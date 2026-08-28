# Local RAG corpus

원문 PDF와 임베딩 데이터는 저장소에 포함하지 않습니다. 아래는 현재 로컬 Qdrant 인덱스의 범위입니다.

| Product | Version/model | Local scope |
| --- | --- | --- |
| RHEL | 9 | LVM, multipath, storage devices, file systems |
| OpenShift | 4.18 | Storage |
| OpenShift | 4.19 | Storage |
| OpenShift | 4.20 | Storage |
| Tesla | Model 3 (2024+) | Owner manual |

외부 최신 정보가 필요한 Claude Code, Grok Build, ChatGPT Codex, 게임, 날씨 질의는 로컬 SearXNG를 통해 공식 출처 우선 검색을 수행합니다. 검색 결과가 없거나 공식 범위를 확인할 수 없으면 답변을 단정하지 않습니다.
