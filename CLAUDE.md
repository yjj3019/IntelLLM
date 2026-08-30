# IntelLLM Claude handoff

## Workspace

- Windows workspace: `C:\AI`
- Git remote: `https://github.com/yjj3019/IntelLLM.git`
- Branch: `main`
- Runtime Python: `C:\AI\npu-env\Scripts\python.exe`
- Project history: [Notion handoff page](https://app.notion.com/p/3c4b44a2dd2e81bd8782d551257c85d8?pvs=204)

## Current known-good state

- FastAPI entrypoint: `server/app.py`; launch with `start-fastapi-v5.1.ps1`.
- Expected `http://127.0.0.1:8000/health`: API `0.12.0`, NPU ready (`Intel AI Boost`, `LFM2-1.2B`), GPU ready (Ollama Vulkan on `Intel Arc 140V`), OCR ready (RapidOCR/OpenVINO/PP-OCRv5, Korean).
- NPU ACL was repaired and direct OpenVINO NPU inference was verified.
- Ollama was recovered with model store `D:\LLM-Model`; installed models include `qwen3:8b`, `qwen3:14b`, and `gemma3:12b`.
- Image-only scanned-PDF smoke validation passed: PDF pages are OCRed and converted into page-aware RAG chunks with provenance.

## Recent changes

- `server/app.py`: optional `LOCAL_AI_API_KEY`, truthful health reporting, streaming disconnect cancellation/socket cleanup, OCR controls, and NPU token metrics.
- `server/rag_engine.py`, `server/web_search.py`: current runtime fixes from the preceding review.
- `rag/scripts/test_scanned_pdf_rag.py`: dependency-light scanned-PDF OCR/RAG smoke test.
- `scripts/benchmark_npu_vs_gpu.py`: deterministic tier benchmark with JSON output.
- `start-fastapi-v5.ps1`: tolerant Qdrant key-loader error handling.
- `start-fastapi-v5.1.ps1`: NPU cold-start wait increased to 300 seconds.

## Validation

```powershell
C:\AI\npu-env\Scripts\python.exe -m py_compile `
  C:\AI\server\rag_engine.py `
  C:\AI\server\web_search.py `
  C:\AI\rag\scripts\ingest_pdf_v3_1.py `
  C:\AI\rag\scripts\search_rag_v4.py

C:\AI\npu-env\Scripts\python.exe C:\AI\rag\scripts\test_scanned_pdf_rag.py
C:\AI\npu-env\Scripts\python.exe C:\AI\scripts\benchmark_npu_vs_gpu.py --help
```

The reduced 12-prompt run measured NPU median `1.0532 s` and GPU median `2.5837 s`; this is not an apples-to-apples model comparison because the tiers use different model sizes. A full multi-run benchmark remains pending.

## Safety and next work

- Never commit or log `LOCAL_AI_API_KEY`, `QDRANT_API_KEY`, DPAPI contents, tokens, or model weights.
- `LOCAL_AI_API_KEY` is opt-in and currently unset; enabling it requires migrating all clients to send the header.
- Before editing, inspect `git status` and the existing implementation. Keep runtime assets, virtual environments, caches, logs, secrets, and RAG data out of Git.
- Candidate next steps: full 5-run NPU/GPU benchmark, persistent Ollama model-store startup configuration, and broader scanned-document/RHEL/OpenShift corpus validation.
