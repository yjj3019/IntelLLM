# IntelLLM

Intel Core Ultra 7 258V용 로컬 AI 라우터입니다. 단순 질의는 OpenVINO NPU로, 기술 질의와 문서 기반 질의는 Intel Arc GPU의 Ollama 모델로 자동 라우팅합니다.

## 현재 기준

- Router: FastAPI `v0.11.14`
- NPU: Intel AI Boost / OpenVINO GenAI / `LFM2-1.2B`
- GPU main: Intel Arc 140V / Ollama Vulkan / `qwen3:8b`
- GPU deep: Ollama / `qwen3:14b` 수동 선택
- RAG: Qdrant + `intfloat/multilingual-e5-base`
- RAG 검색: candidate 12 → deterministic rerank → Top 2
- Live search: 로컬 SearXNG 기반 공식 문서·게임·날씨 검색
- v0.11.14: Apotheosis Minecraft 검색 라우팅·공식 경로 필터, 버전/로더 범위 guard, 게임 검색 반복 방지

## 저장 범위

이 저장소에는 실행 코드와 재현 안내만 포함합니다. 모델 가중치, Python 가상환경, 캐시, Qdrant 데이터, 로그, 원문 PDF, 비밀값은 공개 저장소에 저장하지 않습니다. 현재 로컬 실행본은 `C:\AI` 경로를 기준으로 하므로 다른 환경에서는 `server/app.py`의 모델·캐시 경로를 조정해야 합니다.

## 구성

```text
server/app.py          FastAPI 라우터와 OpenAI 호환 API
server/rag_engine.py   Qdrant 검색, 임베딩, rerank, grounding context
server/web_search.py   SearXNG 기반 실시간 검색과 공식 출처 필터
docs/CORPUS.md         로컬 RAG 인덱스 범위
requirements.txt       Python 의존성
```

## 사전 조건

1. Python 3.11 환경에 `requirements.txt`를 설치합니다.
2. OpenVINO NPU 모델과 캐시를 준비합니다.
3. Ollama에 `qwen3:8b`, 선택적으로 `qwen3:14b`를 준비하고 Vulkan 경로를 활성화합니다.
4. Qdrant를 `http://127.0.0.1:6333`에서 실행합니다.
5. 실시간 검색을 사용하려면 SearXNG를 `http://127.0.0.1:8888`에서 실행합니다.

## 실행

```powershell
python -m uvicorn app:app --app-dir server --host 0.0.0.0 --port 8000
Invoke-RestMethod http://127.0.0.1:8000/health
python test_web_search.py
```

OpenAI 호환 엔드포인트는 `POST /v1/chat/completions`입니다. 모델 이름은 `local-auto`, `local-npu-fast`, `local-gpu-main`, `local-gpu-deep`을 지원합니다.

## 성능 기준

로컬 Intel Core Ultra 7 258V에서 측정한 기준값입니다.

- NPU 워밍업 후 단순 응답: 약 0.6~0.7초
- RHEL RAG 검색: 반복 요청 약 0.14~0.33초
- Qwen3 8B GPU decode: 약 14~16 tok/s
- RHEL RAG 전체 응답: 질문과 출력 길이에 따라 약 14~18초

v0.11.14는 v0.11.13의 NPU/RAG 준비 작업과 Qdrant 연결 재사용을 유지하면서, Apotheosis 질의를 실시간 게임 검색으로 보내고 CurseForge/GitHub의 해당 프로젝트 경로만 공식으로 인정합니다. GPU 모델, RAG 후보 수, Top 2, 문서 발췌 정책은 유지합니다.

## 주의

- NPU 파이프라인의 프롬프트 상한은 1024 토큰입니다. `local-auto`는 이를 넘는 요청을 GPU로 전환합니다.
- 실시간 정보는 검색 결과가 있는 경우에만 답변에 사용하며, 공식 제품·버전 범위가 인덱스에 없으면 추측하지 않습니다.
- 공개 저장소에 모델 파일이나 문서 원문을 추가할 때는 각 파일의 배포 권한을 먼저 확인해야 합니다.
