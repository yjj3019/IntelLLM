"""Per-tier latency benchmark for the local API: NPU fast tier vs GPU main tier.

IMPORTANT -- this is NOT an apples-to-apples comparison.

  local-npu-fast  -> LFM2-1.2B on the Intel AI Boost NPU (OpenVINO)
  local-gpu-main  -> a much larger Qwen3 model on the Arc GPU (Ollama Vulkan)

Different model sizes, different runtimes, different capabilities. The point is
to characterise each tier as it is actually served, so routing decisions can be
made with real numbers -- not to declare a winner.

Standard library only. No credentials are read, stored, or sent.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_LOG_DIR = Path(r"C:\AI\logs")

NPU_MODEL = "local-npu-fast"
GPU_MODEL = "local-gpu-main"
DEFAULT_MODELS = [NPU_MODEL, GPU_MODEL]

DEFAULT_WARMUP = 1
DEFAULT_RUNS = 5
DEFAULT_MAX_TOKENS = 64
DEFAULT_TIMEOUT = 180

# Deterministic, short, and deliberately free of the router's live-info and
# RAG trigger words, so both tiers answer from the model alone. Whether that
# held is verified per call via local_metrics.rag_enabled / live_enabled.
PROMPTS = [
    ("arithmetic", "arith-1", "17 + 25 를 계산해서 숫자만 답해줘."),
    ("arithmetic", "arith-2", "144 를 12 로 나눈 값을 숫자만 답해줘."),
    ("arithmetic", "arith-3", "9 곱하기 7 은 얼마인지 숫자만 답해줘."),

    ("korean_qa", "qa-1", "대한민국의 수도는 어디인지 한 단어로 답해줘."),
    ("korean_qa", "qa-2", "물의 화학식은 무엇인지 기호로만 답해줘."),
    ("korean_qa", "qa-3", "일 년은 몇 개월인지 숫자로만 답해줘."),

    ("classification", "cls-1",
     "다음 문장의 감정을 긍정 또는 부정 중 하나로만 분류해줘: "
     "배송이 정말 빠르고 포장도 깔끔했어요."),
    ("classification", "cls-2",
     "다음 문장의 감정을 긍정 또는 부정 중 하나로만 분류해줘: "
     "화면이 자꾸 꺼져서 너무 불편합니다."),
    ("classification", "cls-3",
     "다음 문장이 질문인지 서술인지 한 단어로만 분류해줘: "
     "회의실 예약은 어떻게 하나요?"),

    ("translation", "trs-1",
     "다음 문장을 영어로 번역해줘: 창문을 닫아 주세요."),
    ("translation", "trs-2",
     "다음 문장을 한국어로 번역해줘: The meeting starts at nine."),
    ("translation", "trs-3",
     "다음 문장을 영어로 번역해줘: 일정을 다시 확인하겠습니다."),
]

CATEGORIES = ["arithmetic", "korean_qa", "classification", "translation"]


class ApiUnavailable(Exception):
    """The local API could not be reached or failed a preflight probe."""


def http_json(url, payload=None, timeout=DEFAULT_TIMEOUT):
    """GET (payload is None) or POST JSON. Returns the decoded body."""
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data else "GET",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")

    return json.loads(body)


def describe_error(error, url):
    if isinstance(error, urllib.error.HTTPError):
        try:
            detail = error.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            detail = ""
        return "HTTP {0} from {1}{2}".format(
            error.code,
            url,
            " -- " + detail if detail else "",
        )

    if isinstance(error, urllib.error.URLError):
        return "cannot reach {0} ({1})".format(url, error.reason)

    return "{0}: {1}".format(type(error).__name__, error)


def preflight(base_url, models, timeout):
    """Confirm the API is up and report which requested models it serves."""
    health_url = base_url + "/health"
    models_url = base_url + "/v1/models"
    probe_timeout = min(timeout, 30)

    try:
        health = http_json(health_url, timeout=probe_timeout)
    except Exception as error:
        raise ApiUnavailable(
            "local API is not available -- "
            + describe_error(error, health_url)
            + "\nStart it first (C:\\AI\\start-fastapi-v5.1.ps1) "
            "or point --base-url at a running instance."
        )

    try:
        served = http_json(models_url, timeout=probe_timeout)
    except Exception as error:
        raise ApiUnavailable(
            "local API answered /health but not /v1/models -- "
            + describe_error(error, models_url)
        )

    available = sorted(
        entry.get("id")
        for entry in served.get("data", [])
        if isinstance(entry, dict) and entry.get("id")
    )

    missing = [model for model in models if model not in available]
    if len(missing) == len(models):
        raise ApiUnavailable(
            "none of the requested models are served: {0} (available: {1})".format(
                ", ".join(models),
                ", ".join(available) or "none",
            )
        )

    return health, available, missing


def call_once(base_url, model, prompt, max_tokens, timeout):
    """One chat completion. Returns a per-call record; never raises."""
    url = base_url + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    started = time.perf_counter()
    try:
        body = http_json(url, payload=payload, timeout=timeout)
    except Exception as error:
        return {
            "ok": False,
            "wall_seconds": round(time.perf_counter() - started, 4),
            "error": describe_error(error, url),
        }

    wall_seconds = round(time.perf_counter() - started, 4)

    choices = body.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content") or ""

    # Usage and local_metrics are recorded exactly as the server returned them.
    # Missing or zero token counts stay missing -- nothing here invents tokens.
    usage = body.get("usage")
    local_metrics = body.get("local_metrics")

    return {
        "ok": True,
        "wall_seconds": wall_seconds,
        "served_model": body.get("model"),
        "response_chars": len(content),
        "response_preview": content.strip().replace("\n", " ")[:120],
        "usage": usage if isinstance(usage, dict) else {},
        "local_metrics": local_metrics if isinstance(local_metrics, dict) else {},
    }


def positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def median_or_none(values):
    numbers = [
        v for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return round(statistics.median(numbers), 4) if numbers else None


def summarise(records):
    """Median-centred summary over a list of successful call records."""
    walls = [record["wall_seconds"] for record in records]

    generation = [
        record["local_metrics"].get("generation_seconds") for record in records
    ]

    completion_tokens = [
        positive_int(record["usage"].get("completion_tokens"))
        for record in records
    ]
    completion_tokens = [t for t in completion_tokens if t is not None]

    throughput = []
    for record in records:
        tokens = positive_int(record["usage"].get("completion_tokens"))
        seconds = record["local_metrics"].get("generation_seconds")
        if not isinstance(seconds, (int, float)) or not seconds:
            seconds = record["wall_seconds"]
        if tokens and seconds:
            throughput.append(tokens / seconds)

    return {
        "calls": len(records),
        "wall_seconds_median": median_or_none(walls),
        "wall_seconds_min": round(min(walls), 4) if walls else None,
        "wall_seconds_max": round(max(walls), 4) if walls else None,
        "generation_seconds_median": median_or_none(generation),
        "completion_tokens_median": median_or_none(completion_tokens),
        "completion_tokens_reported": len(completion_tokens),
        "completion_tokens_per_second_median": median_or_none(throughput),
        "rag_enabled_calls": sum(
            1 for record in records if record["local_metrics"].get("rag_enabled")
        ),
        "live_enabled_calls": sum(
            1 for record in records if record["local_metrics"].get("live_enabled")
        ),
    }


def benchmark_model(base_url, model, args):
    print("\n[{0}] warmup x{1}, measured x{2}, {3} prompts".format(
        model, args.warmup, args.runs, len(PROMPTS)
    ))

    calls = []
    failures = []

    for category, prompt_id, prompt in PROMPTS:
        for _ in range(args.warmup):
            call_once(base_url, model, prompt, args.max_tokens, args.timeout)

        for run in range(1, args.runs + 1):
            record = call_once(
                base_url, model, prompt, args.max_tokens, args.timeout
            )
            record.update(
                {"category": category, "prompt_id": prompt_id, "run": run}
            )

            if record["ok"]:
                calls.append(record)
            else:
                failures.append(record)

        done = [c for c in calls if c["prompt_id"] == prompt_id]
        marker = (
            "{0:.3f}s".format(median_or_none([c["wall_seconds"] for c in done]))
            if done else "FAILED"
        )
        print("  {0:<14} {1:<8} {2}".format(category, prompt_id, marker))

    devices = sorted({
        str(c["local_metrics"].get("device"))
        for c in calls
        if c["local_metrics"].get("device")
    })
    backend_models = sorted({
        str(c["local_metrics"].get("backend_model"))
        for c in calls
        if c["local_metrics"].get("backend_model")
    })

    return {
        "model": model,
        "devices_observed": devices,
        "backend_models_observed": backend_models,
        "failed_calls": len(failures),
        "failures": failures[:10],
        "overall": summarise(calls) if calls else None,
        "by_category": {
            category: summarise(
                [c for c in calls if c["category"] == category]
            )
            for category in CATEGORIES
            if any(c["category"] == category for c in calls)
        },
        "calls": calls,
    }


def print_console_report(report):
    print("\n" + "=" * 70)
    print("PER-TIER BENCHMARK -- NOT apples-to-apples (different model sizes)")
    print("=" * 70)

    header = "{0:<16} {1:>9} {2:>9} {3:>9} {4:>10}".format(
        "model", "median s", "min s", "max s", "tok/s"
    )
    print(header)
    print("-" * len(header))

    for result in report["results"]:
        overall = result["overall"]
        if not overall:
            print("{0:<16} {1}".format(result["model"], "no successful calls"))
            continue

        throughput = overall["completion_tokens_per_second_median"]
        print("{0:<16} {1:>9} {2:>9} {3:>9} {4:>10}".format(
            result["model"],
            overall["wall_seconds_median"],
            overall["wall_seconds_min"],
            overall["wall_seconds_max"],
            "{0:.1f}".format(throughput) if throughput else "n/a",
        ))

    print("\nmedian wall seconds by category")
    row = "{0:<16} {1:>11} {2:>11} {3:>15} {4:>12}"
    print(row.format("model", *CATEGORIES))
    print("-" * 68)

    for result in report["results"]:
        cells = []
        for category in CATEGORIES:
            summary = result["by_category"].get(category)
            cells.append(
                str(summary["wall_seconds_median"]) if summary else "n/a"
            )
        print(row.format(result["model"], *cells))

    for result in report["results"]:
        overall = result["overall"]
        if overall and (
            overall["rag_enabled_calls"] or overall["live_enabled_calls"]
        ):
            print(
                "\n! {0}: {1} call(s) used RAG, {2} used live grounding -- those "
                "timings include retrieval, not just generation.".format(
                    result["model"],
                    overall["rag_enabled_calls"],
                    overall["live_enabled_calls"],
                )
            )
        if overall and not overall["completion_tokens_reported"]:
            print(
                "! {0}: server reported no token counts; tok/s omitted "
                "(not estimated).".format(result["model"])
            )
        if result["failed_calls"]:
            print("! {0}: {1} failed call(s), first: {2}".format(
                result["model"],
                result["failed_calls"],
                result["failures"][0]["error"],
            ))

    if report["models_unavailable"]:
        print("\n! not served by this API, skipped: "
              + ", ".join(report["models_unavailable"]))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local-npu-fast against local-gpu-main on the local API. "
            "Per-tier characterisation, explicitly not apples-to-apples."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="local API base URL (default: %(default)s)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="models to benchmark (default: %(default)s)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="measured runs per prompt (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="warmup runs per prompt, discarded "
                             "(default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="max_tokens per request (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="per-request timeout in seconds "
                             "(default: %(default)s)")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR),
                        help="directory for the JSON log (default: %(default)s)")
    parser.add_argument("--no-log", action="store_true",
                        help="skip writing the JSON log")

    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")

    return args


def main(argv=None):
    args = parse_args(argv)
    base_url = args.base_url.rstrip("/")

    try:
        health, available, missing = preflight(base_url, args.models, args.timeout)
    except ApiUnavailable as error:
        print("ERROR: {0}".format(error), file=sys.stderr)
        return 2

    to_run = [model for model in args.models if model not in missing]

    started_at = datetime.now()
    print("local API {0} -- health: {1}".format(base_url, health.get("status")))
    print("models served: " + ", ".join(available))

    results = [benchmark_model(base_url, model, args) for model in to_run]

    report = {
        "benchmark": "npu_vs_gpu_per_tier",
        "comparison_note": (
            "Per-tier characterisation, NOT apples-to-apples: local-npu-fast is a "
            "1.2B model on the NPU via OpenVINO, local-gpu-main is a much larger "
            "model on the GPU via Ollama Vulkan. Read each tier against its own "
            "routing role, not against the other tier's output quality."
        ),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "health": health,
        "config": {
            "models_requested": args.models,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": False,
            "timeout_seconds": args.timeout,
            "prompt_count": len(PROMPTS),
            "categories": CATEGORIES,
        },
        "models_unavailable": missing,
        "prompts": [
            {"category": category, "id": prompt_id, "prompt": prompt}
            for category, prompt_id, prompt in PROMPTS
        ],
        "results": results,
    }

    print_console_report(report)

    if not args.no_log:
        log_dir = Path(args.log_dir)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "benchmark-{0}.json".format(
                started_at.strftime("%Y%m%d-%H%M%S")
            )
            log_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("\nJSON log: {0}".format(log_path))
        except OSError as error:
            print("\nWARNING: could not write JSON log -- {0}".format(error),
                  file=sys.stderr)

    if all(result["overall"] is None for result in results):
        print("\nERROR: every call failed; no timings produced.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
