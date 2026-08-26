import re
import threading
from typing import List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "redhat_docs"
EMBED_MODEL = "intfloat/multilingual-e5-base"

DEFAULT_TOP_K = 2
DEFAULT_CANDIDATES = 12

RAG_EXCERPT_CHARS = 1400
RAG_EXCERPT_HALF = RAG_EXCERPT_CHARS // 2

_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()
_qdrant_client: Optional[QdrantClient] = None
_qdrant_client_lock = threading.Lock()


# Explicit technical anchors used for lightweight reranking.
# Keep this list conservative: only terms that should matter when the user
# explicitly mentions them or when a Korean phrase maps directly to them.
TECH_ANCHORS = [
    "fast_io_fail_tmo",
    "dev_loss_tmo",
    "eh_deadline",
    "no_path_retry",
    "recovery_tmo",
    "replacement_timeout",
    "multipath",
    "multipathd",
    "iscsi",
    "nvme",
    "nvme/fc",
    "fibre channel",
    "fiber channel",
    "scsi",
    "lvm",
    "raid",
    "stratis",
    "xfs",
    "ext4",
    "lsblk",
    "vmstat",
    "free",
]


INTENT_HINTS = {
    "definition": [
        "의미",
        "뜻",
        "차이",
        "설명",
        "무엇",
        "what",
        "difference",
        "meaning",
    ],
    "removal": [
        "제거",
        "삭제",
        "분리",
        "remove",
        "removal",
        "delete",
        "detach",
    ],
    "failure": [
        "장애",
        "실패",
        "timeout",
        "타임아웃",
        "fail",
        "failure",
        "bad",
    ],
}


def get_embedding_model():
    """Lazy-load the embedding model on the first RAG request."""

    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            print(f"[RAG] Loading embedding model: {EMBED_MODEL}")

            _model = SentenceTransformer(
                EMBED_MODEL,
                device="cpu",
            )

            print("[RAG] Embedding model ready.")

    return _model


def get_qdrant_client() -> QdrantClient:
    """Reuse one local Qdrant client so each RAG query keeps its connection pool."""

    global _qdrant_client

    if _qdrant_client is None:
        with _qdrant_client_lock:
            if _qdrant_client is None:
                _qdrant_client = QdrantClient(
                    url=QDRANT_URL,
                    timeout=30,
                )

    return _qdrant_client


def warm_up_rag() -> None:
    """Load the embedding model and create the reusable Qdrant client."""

    get_embedding_model()
    get_qdrant_client()


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _query_anchors(query: str) -> List[str]:
    q = _normalize(query)
    found = []

    for anchor in TECH_ANCHORS:
        if anchor in q:
            found.append(anchor)

    # Korean wording -> direct technical concepts.
    if (
        "파이버 채널" in q
        or "파이버채널" in q
        or "fibre channel" in q
        or "fiber channel" in q
    ):
        found.extend(
            [
                "fibre channel",
                "fc remote port",
            ]
        )

    if "스토리지 경로" in q or "경로 장애" in q:
        found.extend(
            [
                "path",
                "remote port",
            ]
        )

    if (
        "안전하게 제거" in q
        or (
            "스토리지 장치" in q
            and "제거" in q
        )
    ):
        found.extend(
            [
                "removing storage devices",
                "safe removal",
            ]
        )

    # Preserve order and remove duplicates.
    return list(dict.fromkeys(found))


def _detect_intents(query: str) -> List[str]:
    q = _normalize(query)
    intents = []

    for intent, hints in INTENT_HINTS.items():
        if any(hint in q for hint in hints):
            intents.append(intent)

    return intents


def _rerank_bonus(
    query: str,
    content: str,
) -> Tuple[float, List[str]]:
    """
    Return a small deterministic bonus/penalty applied after vector search.

    This is intentionally lightweight. It does not replace semantic search;
    it only promotes chunks that contain the exact technical concepts and
    explanatory patterns requested by the user.
    """

    text = _normalize(content)
    anchors = _query_anchors(query)
    intents = _detect_intents(query)

    bonus = 0.0
    reasons: List[str] = []

    matched_anchors = [
        anchor
        for anchor in anchors
        if anchor in text
    ]

    if matched_anchors:
        anchor_bonus = min(
            0.045,
            0.015 * len(matched_anchors),
        )
        bonus += anchor_bonus

        reasons.append(
            "anchors="
            + ",".join(matched_anchors[:5])
        )

    if "definition" in intents:
        definition_patterns = [
            "the number of seconds",
            "specifies the number of seconds",
            "controls when",
            "default value",
            "setting this",
        ]

        hits = sum(
            pattern in text
            for pattern in definition_patterns
        )

        if hits:
            bonus += min(
                0.025,
                0.007 * hits,
            )
            reasons.append(
                f"definition_hits={hits}"
            )

    if "removal" in intents:
        removal_patterns = [
            "removing storage devices",
            "safe removal of storage devices",
            "safely remove",
            "remove a storage device",
            "/device/delete",
            "lsblk",
            "top-to-bottom",
        ]

        hits = sum(
            pattern in text
            for pattern in removal_patterns
        )

        if hits:
            bonus += min(
                0.035,
                0.008 * hits,
            )
            reasons.append(
                f"removal_hits={hits}"
            )

    if "failure" in intents:
        failure_patterns = [
            "dev_loss_tmo",
            "fast_io_fail_tmo",
            "remote port",
            "failing i/o",
            "device is removed",
            'marks a link as "bad"',
        ]

        hits = sum(
            pattern in text
            for pattern in failure_patterns
        )

        if hits:
            bonus += min(
                0.035,
                0.007 * hits,
            )
            reasons.append(
                f"failure_hits={hits}"
            )

    # Penalize obvious table-of-contents-like chunks.
    toc_signals = [
        "table of contents",
        ". . . . . . .",
    ]

    if any(
        signal in text
        for signal in toc_signals
    ):
        bonus -= 0.035
        reasons.append("toc_penalty")

    # Small penalty for very sparse/list-like chunks.
    alpha_count = len(
        re.findall(
            r"[a-zA-Z가-힣]",
            content,
        )
    )
    line_count = max(
        1,
        len(content.splitlines()),
    )

    if (
        line_count > 25
        and alpha_count / line_count < 18
    ):
        bonus -= 0.015
        reasons.append("sparse_penalty")

    return bonus, reasons


def search_redhat_docs(
    query: str,
    product: str = "RHEL",
    version: str = "9",
    limit: int = DEFAULT_TOP_K,
    candidates: int = DEFAULT_CANDIDATES,
):
    """
    Semantic vector search followed by deterministic technical reranking.

    The FastAPI app normally requests limit=2, while Qdrant retrieves a
    broader candidate pool so that misleading near-neighbours (for example
    NVMe/FC setup chunks for an FC timeout question) can be pushed down.
    """

    model = get_embedding_model()

    client = get_qdrant_client()

    vector = model.encode(
        "query: " + query,
        normalize_embeddings=True,
    ).tolist()

    query_filter = Filter(
        must=[
            FieldCondition(
                key="product",
                match=MatchValue(
                    value=product
                ),
            ),
            FieldCondition(
                key="version",
                match=MatchValue(
                    value=version
                ),
            ),
        ]
    )

    candidate_count = max(
        limit,
        candidates,
    )

    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=candidate_count,
        with_payload=True,
        with_vectors=False,
    ).points

    reranked = []

    for result in results:
        payload = result.payload or {}
        content = payload.get(
            "content",
            "",
        )

        bonus, reasons = _rerank_bonus(
            query=query,
            content=content,
        )

        vector_score = float(
            result.score
        )
        final_score = (
            vector_score
            + bonus
        )

        reranked.append(
            {
                "result": result,
                "vector_score": vector_score,
                "rerank_bonus": bonus,
                "final_score": final_score,
                "rerank_reasons": reasons,
            }
        )

    reranked.sort(
        key=lambda item: item[
            "final_score"
        ],
        reverse=True,
    )

    documents = []

    for item in reranked[:limit]:
        result = item["result"]
        payload = result.payload or {}

        documents.append(
            {
                # score remains the score used for final ranking.
                "score":
                    item["final_score"],

                "vector_score":
                    item["vector_score"],

                "rerank_bonus":
                    item["rerank_bonus"],

                "rerank_reasons":
                    item["rerank_reasons"],

                "product":
                    payload.get(
                        "product",
                        "",
                    ),

                "version":
                    payload.get(
                        "version",
                        "",
                    ),

                "title":
                    payload.get(
                        "title",
                        "",
                    ),

                "source_url":
                    payload.get(
                        "source_url",
                        "",
                    ),

                "source_file":
                    payload.get(
                        "source_file",
                        "",
                    ),

                "document_id":
                    payload.get(
                        "document_id",
                        "",
                    ),

                "page_start":
                    payload.get(
                        "page_start"
                    ),

                "page_end":
                    payload.get(
                        "page_end"
                    ),

                "chunk_index":
                    payload.get(
                        "chunk_index"
                    ),

                "content":
                    payload.get(
                        "content",
                        "",
                    ),
            }
        )

    if documents:
        summary = []

        for index, doc in enumerate(
            documents,
            start=1,
        ):
            summary.append(
                (
                    f"#{index} "
                    f"{doc['title']} "
                    f"vector={doc['vector_score']:.4f} "
                    f"bonus={doc['rerank_bonus']:+.4f} "
                    f"final={doc['score']:.4f}"
                )
            )

        print(
            "[RAG] Reranked: "
            + " | ".join(summary)
        )

    return documents


def _mark_extraction_gaps(
    content: str,
) -> str:
    """
    Mark obvious PDF text-extraction holes so the LLM does not invent
    missing numeric defaults or limits.

    This only changes the context sent to the model; stored Qdrant
    payloads remain untouched.
    """

    text = content

    replacements = [
        (
            r"(?i)\bdefault value is\s*\.",
            "default value is [[EXTRACTION_GAP_NUMERIC_VALUE]].",
        ),
        (
            r"(?i)\bcapped to\s+seconds\b",
            "capped to [[EXTRACTION_GAP_NUMERIC_VALUE]] seconds",
        ),
        (
            r"(?i)\bis set to\s+seconds\b",
            "is set to [[EXTRACTION_GAP_NUMERIC_VALUE]] seconds",
        ),
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
        )

    return text


def _compact_doc_excerpt(
    content: str,
    query: str,
    max_chars: int = RAG_EXCERPT_CHARS,
) -> str:
    """
    Keep only a query-focused window from a retrieved chunk.

    This reduces prompt size and repetitive adjacent text while preserving
    the original Qdrant payload. The excerpt is centered on the earliest
    query anchor found in the chunk.
    """

    text = _mark_extraction_gaps(
        content
    ).strip()

    if len(text) <= max_chars:
        return text

    lowered = text.lower()

    positions = []

    for anchor in _query_anchors(query):
        pos = lowered.find(
            anchor.lower()
        )

        if pos >= 0:
            positions.append(pos)

    if positions:
        center = min(positions)

        start = max(
            0,
            center - RAG_EXCERPT_HALF,
        )

        end = min(
            len(text),
            start + max_chars,
        )

        start = max(
            0,
            end - max_chars,
        )

        excerpt = text[
            start:end
        ].strip()

        if start > 0:
            excerpt = "... " + excerpt

        if end < len(text):
            excerpt = excerpt + " ..."

        return excerpt

    return (
        text[:max_chars].strip()
        + " ..."
    )


def build_rag_context(
    query: str,
    product: str = "RHEL",
    version: str = "9",
    limit: int = DEFAULT_TOP_K,
):
    docs = search_redhat_docs(
        query=query,
        product=product,
        version=version,
        limit=limit,
    )

    if not docs:
        return "", []

    if product == "RHEL":
        scope_rule = (
            "Keep parameters separate and preserve before/after/until "
            "direction exactly. For FC timeouts, say that fast_io_fail_tmo "
            "causes I/O failure only after its timeout expires, not merely "
            "when configured. Say that blocked-queue I/O may wait until "
            "dev_loss_tmo only when that condition is supported; do not "
            "generalize it to all I/O. When both parameters are asked, "
            "preserve this sequence: FC remote-port failure detection, "
            "fast_io_fail_tmo expiry and path I/O failure, then "
            "dev_loss_tmo expiry and remote-port/device removal. Do not "
            "say that configuring a timeout immediately fails I/O. If "
            "fast_io_fail_tmo is numeric, running or new path I/O fails "
            "when that timeout triggers; only I/O in a blocked queue waits "
            "until dev_loss_tmo expires and the queue unblocks. If "
            "fast_io_fail_tmo is off, no I/O fails until device removal. "
            "Do not describe dev_loss_tmo as the general I/O-failure timer."
        )
        final_rule = (
            "Use only the excerpts, do not invent missing numbers or "
            "example values, state timeout expiry before I/O failure, "
            "do not say all I/O waits until dev_loss_tmo, and keep the "
            "answer short enough to finish naturally."
        )
    else:
        scope_rule = (
            "Preserve the product, model, version, region, and software "
            "scope stated by the sources. If the source does not cover the "
            "asked scope, say so instead of generalizing it. Treat exact "
            "commands, endpoints, limits, and feature availability as "
            "verified only when the excerpts state them explicitly."
        )
        final_rule = (
            "Use only the excerpts, do not invent missing values, and do "
            "not extend one product, model, version, or region to another."
        )

    parts = [
        (
            f"Use only the supplied official {product}/{version} excerpts. "
            "Do not use unsupported model knowledge."
        ),
        (
            "For Korean questions, answer in natural Korean."
        ),
        (
            "Never guess missing numeric values or invent example values. "
            "If [[EXTRACTION_GAP_NUMERIC_VALUE]] appears, say the exact "
            "number is unavailable in the extracted text."
        ),
        scope_rule,
        (
            "Do not rewrite infinity as never; use the concrete source "
            "representation when available."
        ),
        (
            "Be concise: for two parameters, use two short bullets and one "
            "short interaction paragraph. Avoid repetition and secondary "
            "details unless required."
        ),
        "",
        f"OFFICIAL {product.upper()} EXCERPTS:",
        "",
    ]

    for index, doc in enumerate(
        docs,
        start=1,
    ):
        page_text = ""

        if doc["page_start"] is not None:
            if (
                doc["page_end"] is not None
                and doc["page_end"]
                != doc["page_start"]
            ):
                page_text = (
                    f"pages {doc['page_start']}"
                    f"-{doc['page_end']}"
                )
            else:
                page_text = (
                    f"page {doc['page_start']}"
                )

        parts.append(
            f"[SOURCE {index}] "
            f"{doc['title']}"
            + (
                f" ({page_text})"
                if page_text
                else ""
            )
        )

        excerpt = _compact_doc_excerpt(
            doc["content"],
            query=query,
        )

        parts.append(
            excerpt
        )

        parts.append("")

    parts.extend(
        [
            "FINAL CHECK:",
            final_rule,
        ]
    )

    return "\n".join(parts), docs
