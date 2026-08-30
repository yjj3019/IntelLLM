# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local-first LLM router running on a single Windows PC (Intel NPU + Arc GPU), fronted by an
OpenAI-compatible FastAPI service. It routes chat requests to one of three backends based on
prompt heuristics, augments answers with a Qdrant-backed RAG corpus (Red Hat/OpenShift docs) and
live web search, and can OCR scanned PDFs/images into RAG chunks.

- Windows workspace: `C:\AI`
- Git remote: `https://github.com/yjj3019/IntelLLM.git`, branch `main`
- Runtime Python: `C:\AI\npu-env\Scripts\python.exe` (project virtualenv; do not use system Python)
- Project history: [Notion handoff page](https://app.notion.com/p/3c4b44a2dd2e81bd8782d551257c85d8?pvs=204)

## Commands

Start the API (idempotent — checks port 8000 first, retries NPU cold start up to 300s):

```powershell
powershell -ExecutionPolicy Bypass -File C:\AI\start-fastapi-v5.1.ps1
```

This loads the Qdrant API key (`load-qdrant-api-key.ps1`), sets `NO_PROXY` for the NAS LAN and
`HF_HUB_OFFLINE=1`, then runs `uvicorn app:app --app-dir C:\AI\server --host 0.0.0.0 --port 8000`.
Logs land in `C:\AI\logs\fastapi-{stdout,stderr,runner}-v5.1.log`.

Compile-check after editing server/RAG code (there is no formal test suite):

```powershell
C:\AI\npu-env\Scripts\python.exe -m py_compile `
  C:\AI\server\rag_engine.py `
  C:\AI\server\web_search.py `
  C:\AI\rag\scripts\ingest_pdf_v3_1.py `
  C:\AI\rag\scripts\search_rag_v4.py
```

Smoke tests (scripts under `rag/scripts/` and `scripts/` named `test_*.py` are ad hoc runnable
checks, not a pytest suite — run them directly):

```powershell
C:\AI\npu-env\Scripts\python.exe C:\AI\rag\scripts\test_scanned_pdf_rag.py
C:\AI\npu-env\Scripts\python.exe C:\AI\scripts\benchmark_npu_vs_gpu.py --help
```

Health check once running: `GET http://127.0.0.1:8000/health` — expect NPU ready
(`Intel AI Boost`, `LFM2-1.2B`), GPU ready (Ollama Vulkan on `Intel Arc 140V`), and OCR ready
(RapidOCR/OpenVINO/PP-OCRv5, Korean).

## Architecture

Everything the API needs lives in `server/`:

- `app.py` — the FastAPI app and router. `choose_model()` picks the backend per request:
  `local-npu-fast` (LFM2-1.2B via OpenVINO GenAI, short arithmetic/simple patterns only),
  `local-gpu-main` (Ollama `qwen3:8b`, the default for everything else — tech keywords, code
  blocks, long prompts, live-info queries), or `local-gpu-deep` (`qwen3:14b`, opt-in only). It
  also implements the OpenAI-compatible `/v1/chat/completions` streaming endpoint, `/v1/ocr`,
  `/health`, `/v1/models`, and optional `x-api-key`/Bearer auth via `LOCAL_AI_API_KEY` (unset =
  open, per `AGENTS.md`).
- `rag_engine.py` — Qdrant search and reranking (`search_redhat_docs`, `build_rag_context`) over
  the Red Hat/OpenShift corpus, plus embedding-model warm-up.
- `web_search.py` — live grounding: SearXNG-backed search, weather/time/game-profile detection,
  official-domain matching, called from `app.py` when `is_live_query()` fires.
- `ocr_engine.py` — RapidOCR/OpenVINO document parsing (`parse_document`) for scanned PDFs and
  images, producing page-aware chunks with provenance for ingestion.

`rag/scripts/` holds standalone ingestion/inspection CLIs (`ingest_pdf_v3_1.py`,
`search_rag_v4.py`, etc.) that talk to the same Qdrant collection directly — **numeric suffixes
are version history, not variants**: always use the highest-numbered script, ignore older ones
unless explicitly asked to diff behavior. The same pattern applies throughout `server/` and
`scripts/` (`app-v0.11.16.py`, `rag_engine-v0.11.10-backup.py`, etc.) — these are backups kept
for rollback, not alternatives to edit. Only `server/app.py`, `server/rag_engine.py`,
`server/web_search.py`, and `server/ocr_engine.py` (no version suffix) are live.

`scripts/` holds NPU/GPU benchmark and hardware-capability probes (`benchmark_npu_vs_gpu.py`,
`test_*_npu.py`, `test_gpu*.py`) run manually, not part of any CI.

## External services (NAS network)

- Qdrant: `http://192.168.1.3:6333` (override with `QDRANT_URL`); API key decrypted at process
  start from `C:\AI\secrets\qdrant-api-key.dpapi` via Windows DPAPI — never available in plaintext
  on disk. For manual scripts, dot-source `load-qdrant-api-key.ps1` first.
- SearXNG: `http://192.168.1.3:8888` (override with `LOCAL_SEARCH_URL`).
- OpenWebUI: `http://192.168.1.3:3080`, points at this service's OpenAI-compatible base URL,
  `http://192.168.0.112:8000/v1` from the NAS network.
- Ollama model store: `D:\LLM-Model` (models: `qwen3:8b`, `qwen3:14b`, `gemma3:12b`).

## Conventions specific to this repo

- Never commit or log `LOCAL_AI_API_KEY`, `QDRANT_API_KEY`, DPAPI contents, tokens, or model
  weights. `secrets/`, `models/`, `logs/`, `cache/`, `npu-env/` are runtime/local-only.
- When editing routing or RAG-trigger logic in `app.py`, check both Korean and English keyword
  lists (`TECH_KEYWORDS`, `SIMPLE_PATTERNS`) — the router and comments mix both languages.
- Before editing, check `git status` and read the existing implementation of the *unsuffixed*
  file — don't assume the newest-looking backup file is current without checking timestamps.
