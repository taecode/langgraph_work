# LangGraph Work - 프로젝트 컨텍스트

## 프로젝트 개요
LangGraph를 활용한 RAG 시스템 학습 및 실습 프로젝트.

## 환경
- Python 3.12 (uv로 패키지 관리)
- 가상환경: `.venv`
- 주요 라이브러리: `langgraph`, `langchain`, `langchain-openai`, `langchain-pinecone`, `pinecone`
- `.env` 파일에 `OPENAI_API_KEY`, `PINECONE_API_KEY` 저장

## LLM / 임베딩 설정
- **LLM**: `gpt-4o-mini`, `temperature=0`
- **Embedding**: `text-embedding-3-large` (3072차원)
- **VectorDB**: Pinecone Serverless (aws/us-east-1)

## 파일 구조

| 파일 | 설명 |
|------|------|
| `01_langgraph.ipynb` | LangGraph 기초 실습 |
| `02_langgraph.ipynb` | LangGraph 심화 실습 |
| `03_langgraph.ipynb` | LangGraph 추가 실습 |
| `03_langgraph_설명.md` | 03 노트북 단계별 설명 |
| `04_tax_graph.ipynb` | 세법 Self-RAG 그래프 |
| `04_tax_graph_설명.md` | 04 노트북 단계별 설명 |
| `05_privacy_law_graph.ipynb` | 개인정보보호법 Self-RAG 그래프 |
| `06_privacy_law_multiagent.ipynb` | 개인정보보호법 Multi-Agent Architecture |
| `data/tax.docx` | 세법 원본 문서 |
| `data/privacy_law.txt` | 개인정보보호법 조문 텍스트 (casenote.kr 수집) |
| `chroma_db/` | 로컬 벡터 DB (Chroma, 초기 실습용) |

---

## 04_tax_graph.ipynb - 세법 Self-RAG 그래프

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
| `generate` | 노드 | 문서 기반 답변 생성 |
| `check_hallucination` | 조건부 엣지 | 환각 여부 판단 → not_hallucinated / generate |

### 주요 설계 포인트
- **벡터 DB**: Pinecone, index명 `tax-index`
- **Self-RAG 패턴**: 문서 관련성 + 환각 여부를 그래프 노드로 명시적 검증

### AgentState
```python
class AgentState(TypedDict):
    query   : str            # 사용자 질문
    context : List[Document] # 검색된 문서
    answer  : str            # 생성된 답변
```

---

## 05_privacy_law_graph.ipynb - 개인정보보호법 Self-RAG 그래프

04_tax_graph와 동일한 Self-RAG 패턴, 주제만 변경.

### 주요 변경점 (vs 04)
- **문서**: `data/privacy_law.txt` (TextLoader 사용)
- **Pinecone index**: `privacy-index`
- **용어 사전**: 개인정보보호법 전용 (정보주체, 개인정보처리자 등)
- **청킹**: RecursiveCharacterTextSplitter, chunk_size=800

### 그래프 흐름
04_tax_graph와 동일 구조.

---

## 06_privacy_law_multiagent.ipynb - Multi-Agent Architecture

05_privacy_law_graph를 Supervisor + Specialist Agents 패턴으로 발전.

### 그래프 흐름
```
START → supervisor → [전문 에이전트 선택]
          ├── collection  (제15~18조: 수집·이용·제3자 제공)
          ├── sensitive   (제23~24조: 민감정보·고유식별정보)
          ├── security    (제29~31조·34조: 안전관리·유출신고)
          ├── rights      (제35~39조: 정보주체 권리·손해배상)
          └── general     (제1~5조: 총칙·원칙)
                ↓
          quality_checker (환각 검증)
                ├── not hallucinated → END
                └── hallucinated → 동일 에이전트 재실행 (최대 2회)
```

### 노드 구성

| 노드 | 역할 |
|------|------|
| `supervisor` | Structured Output으로 5개 에이전트 중 라우팅 결정 |
| `collection_agent` | 수집·이용·제3자 제공 전문 답변 생성 |
| `sensitive_agent` | 민감정보·고유식별정보 전문 답변 생성 |
| `security_agent` | 안전관리·유출신고 전문 답변 생성 |
| `rights_agent` | 정보주체 권리·손해배상 전문 답변 생성 |
| `general_agent` | 총칙·원칙 전문 답변 생성 |
| `quality_checker` | 환각 여부 검증, 재시도 카운트 관리 |

### MultiAgentState
```python
class MultiAgentState(TypedDict):
    query          : str            # 사용자 질문
    selected_agent : str            # supervisor 라우팅 결정
    context        : List[Document] # 검색된 문서
    answer         : str            # 에이전트 생성 답변
    retry_count    : int            # 환각 재시도 횟수 (max 2)
```

### Single vs Multi-Agent 비교

| 항목 | Single (05) | Multi-Agent (06) |
|------|-------------|-----------------|
| 라우팅 | 없음 | Supervisor가 전문가 선택 |
| 답변 프롬프트 | 범용 RAG prompt | 도메인 특화 system prompt |
| 확장성 | 노드 추가 복잡 | 에이전트 추가/교체 용이 |

### 주의사항
- 그래프 시각화: `MermaidDrawMethod.PYPPETEER` 사용 (외부 mermaid.ink API 타임아웃 문제)

---

## 실행 예시

```python
# Single Agent (05)
result = graph.invoke({'query': '개인정보를 제3자에게 제공할 때 필요한 절차는?'})
print(result['answer'].content)

# Multi-Agent (06)
result = graph.invoke({'query': '고객 정보가 유출됐을 때 회사는 무엇을 해야 하나요?', 'retry_count': 0})
print(result['answer'])
```
