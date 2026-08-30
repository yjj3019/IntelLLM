"""Acceptance smoke test: multi-page image-only PDF -> OCR -> page-aware RAG chunks.

Self-contained: builds the fixture PDF in memory, forces OCR, and exercises the
v3.1 ingest chunking helpers. No network, no Qdrant, no secrets.
"""

from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile

import pymupdf
from PIL import Image, ImageDraw, ImageFont


SERVER_DIR = Path(__file__).resolve().parents[2] / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from ocr_engine import parse_document

RAG_SCRIPT_DIR = Path(__file__).resolve().parent
if str(RAG_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_SCRIPT_DIR))

from ingest_pdf_v3_1 import (
    build_document_and_page_spans,
    extract_ocr_pages,
    make_chunks,
)

PAGE_TEXTS = (
    "ALPHA STORAGE TIMEOUT 30",
    "BRAVO NETWORK RETRY 45",
)
PAGE_MARKERS = ("ALPHA", "BRAVO")


def render_page_image(text: str) -> bytes:
    image = Image.new("RGB", (1200, 180), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)
    draw.text((24, 48), text, fill="black", font=font)
    payload = BytesIO()
    image.save(payload, format="PNG")
    return payload.getvalue()


def build_image_only_pdf() -> bytes:
    document = pymupdf.open()
    for text in PAGE_TEXTS:
        page = document.new_page(width=900, height=135)
        page.insert_image(page.rect, stream=render_page_image(text))
    data = document.tobytes()
    document.close()

    # The fixture must be image-only, otherwise OCR is not what is under test.
    probe = pymupdf.open(stream=data, filetype="pdf")
    try:
        assert len(probe) == 2
        for page in probe:
            assert not (page.get_text("text") or "").strip()
    finally:
        probe.close()
    return data


def main() -> None:
    pdf_data = build_image_only_pdf()

    parsed = parse_document(pdf_data, "scanned-two-page.pdf", "english", force_ocr=True)

    assert parsed["format"] == "pdf"
    assert parsed["engine"] == "RapidOCR"
    assert parsed["language"] == "english"
    assert parsed["page_count"] == 2
    assert parsed["ocr_pages"] == 2
    assert parsed["native_pages"] == 0
    assert parsed["empty_pages"] == 0
    assert len(parsed["pages"]) == 2

    # Page order and per-page provenance.
    assert [page["page_number"] for page in parsed["pages"]] == [1, 2]
    for page, marker in zip(parsed["pages"], PAGE_MARKERS):
        assert page["source"] == "ocr", page
        assert page["blocks"], page
        assert marker in page["text"].upper(), page["text"]

    # No cross-page bleed: each marker appears on its own page only.
    assert "BRAVO" not in parsed["pages"][0]["text"].upper()
    assert "ALPHA" not in parsed["pages"][1]["text"].upper()

    # Document text keeps both pages, in order.
    document_text = parsed["text"].upper()
    assert document_text.index("ALPHA") < document_text.index("BRAVO")

    file_handle, file_name = tempfile.mkstemp(suffix=".pdf")
    os.close(file_handle)
    temp_path = Path(file_name)
    try:
        temp_path.write_bytes(pdf_data)
        rag_pages, rag_result = extract_ocr_pages(temp_path, "english", True)
        rag_document, rag_spans = build_document_and_page_spans(rag_pages)
        rag_chunks = make_chunks(rag_document, rag_spans)
    finally:
        temp_path.unlink(missing_ok=True)

    assert rag_result["page_count"] == 2
    assert rag_result["ocr_pages"] == 2
    assert [page_number for page_number, _ in rag_pages] == [1, 2]
    assert len(rag_spans) == 2
    assert [span[2] for span in rag_spans] == [1, 2]

    assert rag_chunks
    for chunk in rag_chunks:
        assert chunk["page_start"] is not None, chunk
        assert chunk["page_end"] is not None, chunk
        assert 1 <= chunk["page_start"] <= chunk["page_end"] <= 2, chunk

    covered = set()
    for chunk in rag_chunks:
        covered.update(range(chunk["page_start"], chunk["page_end"] + 1))
    assert covered == {1, 2}, covered

    chunk_text = " ".join(chunk["content"] for chunk in rag_chunks).upper()
    for marker in PAGE_MARKERS:
        assert marker in chunk_text, chunk_text

    # Each marker must be reachable through a chunk whose page range covers its page.
    for page_number, marker in enumerate(PAGE_MARKERS, start=1):
        assert any(
            marker in chunk["content"].upper()
            and chunk["page_start"] <= page_number <= chunk["page_end"]
            for chunk in rag_chunks
        ), marker

    print("SCANNED_PDF_RAG_OK", parsed["ocr_pages"], len(rag_chunks))


if __name__ == "__main__":
    main()
