# LangGraph Work - 프로젝트 컨텍스트

## 프로젝트 개요
LangGraph를 활용한 RAG 시스템 학습 및 실습 프로젝트.

## 환경
- Python (uv로 패키지 관리)
- 가상환경: `.venv`
- 주요 라이브러리: `langgraph`, `langchain`, `langchain-openai`, `langchain-pinecone`, `pinecone`
- `.env` 파일에 `OPENAI_API_KEY`, `PINECONE_API_KEY` 저장

## 파일 구조

| 파일 | 설명 |
|------|------|
| `01_langgraph.ipynb` | LangGraph 기초 실습 |
| `02_langgraph.ipynb` | LangGraph 심화 실습 |
| `03_langgraph.ipynb` | LangGraph 추가 실습 |
| `03_langgraph_설명.md` | 03 노트북 단계별 설명 |
| `04_tax_graph.ipynb` | 세법 RAG 시스템 (메인 작업) |
| `04_tax_graph_설명.md` | 04 노트북 단계별 설명 |
| `data/tax.docx` | 세법 원본 문서 |
| `chroma_db/` | 로컬 벡터 DB (Chroma) |

## 04_tax_graph.ipynb - 세법 RAG 그래프

### 그래프 흐름
```
START → retrieve → check_doc_relevence
                        │              │
                     generate       rewrite → retrieve (루프)
                        │
                   check_hallucination
                        │                    │
                 not_hallucinated          generate (재생성 루프)
                        │
                       END
```

### 노드 구성

| 노드 / 엣지 | 종류 | 역할 |
|---|---|---|
| `retrieve` | 노드 | Pinecone에서 관련 세법 문서 검색 (k=2) |
| `check_doc_relevence` | 조건부 엣지 | 문서 관련성 판단 → generate / rewrite |
| `rewrite` | 노드 | 질문을 세법 용어로 재작성 후 재검색 |
| `generate` | 노드 | 문서 기반 GPT-4o-mini 답변 생성 |
| `check_hallucination` | 조건부 엣지 | 환각 여부 판단 → not_hallucinated / generate |

### 주요 설계 포인트
- **벡터 DB**: Pinecone, index명 `tax-index`, 임베딩 모델 `text-embedding-3-large`
- **LLM**: `gpt-4o-mini`, `temperature=0`
- **구조화된 출력**: `with_structured_output()`으로 조건부 엣지 판단을 `yes/no`로 강제
- **Self-RAG 패턴**: 문서 관련성 + 환각 여부를 그래프 노드로 명시적 검증

### AgentState
```python
class AgentState(TypedDict):
    query   : str            # 사용자 질문
    context : List[Document] # 검색된 문서
    answer  : str            # 생성된 답변
```

### 실행 예시
```python
result = graph.invoke({'query': '연봉 5천만원 거주자의 소득세는 얼마인가요'})
print(result['answer'])
```
