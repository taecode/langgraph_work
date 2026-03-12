# 04_tax_graph.ipynb 단계별 설명

세법 문서를 기반으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 시스템을 LangGraph로 구현한 노트북입니다.

---

## 전체 그래프 흐름

```
START
  │
  ▼
retrieve ──────────────────────────────────┐
  │                                        │
  ▼ (check_doc_relevence)                  │
관련 있음? ──── 없음 ──→ rewrite ──────────┘
  │
  ▼ (있음)
generate
  │
  ▼ (check_hallucination)
환각 없음? ──── 있음 ──→ generate (재생성)
  │
  ▼ (없음)
END
```

---

## 1단계: 환경 설정

```python
from dotenv import load_dotenv
load_dotenv()
```

`.env` 파일에서 `OPENAI_API_KEY`, `PINECONE_API_KEY` 등 환경변수를 불러옵니다.

---

## 2단계: 임베딩 모델 설정

```python
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
```

텍스트를 벡터로 변환하는 OpenAI 임베딩 모델을 초기화합니다.
`text-embedding-3-large`는 OpenAI의 고성능 임베딩 모델로 검색 정확도가 높습니다.

---

## 3단계: 벡터 DB 연결

```python
vectorstore = PineconeVectorStore(index_name='tax-index', embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
```

Pinecone에 미리 저장된 세법 문서 벡터 DB(`tax-index`)에 연결합니다.
`k=2`로 설정해 질문과 가장 유사한 문서 2개를 검색합니다.

---

## 4단계: 상태(State) 정의

```python
class AgentState(TypedDict):
    query   : str            # 사용자 질문
    context : List[Document] # 검색된 문서 목록
    answer  : str            # 최종 답변
```

LangGraph에서 노드 간에 공유되는 상태 객체입니다.
각 노드는 이 상태를 읽고 필요한 키만 업데이트해서 반환합니다.

---

## 5단계: retrieve 노드

```python
def retrieve(state: AgentState) -> AgentState:
    docs = retriever.invoke(state['query'])
    return {'context': docs}
```

**역할:** 사용자의 질문(`query`)으로 벡터 DB를 검색해 관련 문서를 가져옵니다.

- 입력: `state['query']`
- 출력: `state['context']` (검색된 문서 리스트)

---

## 6단계: check_doc_relevence (조건부 엣지)

```python
def check_doc_relevence(state: AgentState) -> Literal['generate', 'rewrite']:
    result = chain.invoke({'context': context_text, 'query': query})
    return 'generate' if result.score == 'yes' else 'rewrite'
```

**역할:** 검색된 문서가 질문에 답하기에 충분한지 LLM이 판단합니다.

- 관련 있음(`yes`) → `generate` 노드로 이동
- 관련 없음(`no`) → `rewrite` 노드로 이동

`GradeDocuments` Pydantic 모델과 `with_structured_output()`으로 LLM 출력을 `yes/no`로 강제합니다.

---

## 7단계: rewrite 노드

```python
def rewrite(state: AgentState) -> AgentState:
    new_query = chain.invoke({'query': query})
    return {'query': new_query}
```

**역할:** 검색 결과가 부적절할 때 질문을 세법 용어에 맞게 재작성합니다.

- 예: `"연봉 5천 세금"` → `"연간 총급여 5,000만원 거주자의 종합소득세 세율 및 산출세액"`
- 재작성된 질문으로 다시 `retrieve` 노드를 실행합니다.

---

## 8단계: generate 노드

```python
def generate(state: AgentState) -> AgentState:
    answer = chain.invoke({'context': context_text, 'query': query})
    return {'answer': answer}
```

**역할:** 검색된 문서(`context`)를 바탕으로 GPT-4o-mini가 답변을 생성합니다.

- 시스템 프롬프트로 "세법 전문가" 역할 부여
- 입력: `state['query']`, `state['context']`
- 출력: `state['answer']`

---

## 9단계: check_hallucination (조건부 엣지)

```python
def check_hallucination(state: AgentState) -> Literal['useful', 'generate']:
    result = chain.invoke({'context': context_text, 'answer': answer})
    return 'useful' if result.score == 'yes' else 'generate'
```

**역할:** 생성된 답변이 실제로 문서에 근거하는지 LLM이 검증합니다.

- 문서 근거 있음(`yes`) → `not_hallucinated` → `END`로 이동해 최종 답변 반환
- 환각 감지(`no`) → `generate` 노드로 돌아가 답변 재생성

`GradeHallucination` Pydantic 모델로 판단 결과를 `yes/no`로 구조화합니다.

> **환각(Hallucination)이란?**
> LLM이 문서에 없는 내용을 마치 사실인 것처럼 생성하는 현상입니다.
> 이 단계가 그래프에 명시적으로 존재함으로써 답변 품질을 자동으로 보장합니다.

---

## 10단계: 그래프 구성 및 실행

```python
graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges('retrieve', check_doc_relevence, {...})
graph_builder.add_edge('rewrite', 'retrieve')
graph_builder.add_conditional_edges('generate', check_hallucination, {...})
```

**엣지 종류:**
| 엣지 | 타입 | 설명 |
|------|------|------|
| `START → retrieve` | 일반 | 항상 검색부터 시작 |
| `retrieve → ?` | 조건부 | 문서 관련성 판단 후 분기 |
| `rewrite → retrieve` | 일반 | 질문 재작성 후 다시 검색 (루프) |
| `generate → ?` | 조건부 | 환각 여부 판단 후 분기 |
| `generate → END` | 조건부 | `not_hallucinated`일 때만 종료 |

---

## 전체 노드 역할 요약

| 노드 | 종류 | 역할 |
|------|------|------|
| `retrieve` | 일반 노드 | 벡터 DB에서 관련 문서 검색 |
| `check_doc_relevence` | 조건부 엣지 | 문서 관련성 판단 → generate / rewrite 분기 |
| `rewrite` | 일반 노드 | 질문을 세법 용어로 재작성 |
| `generate` | 일반 노드 | 문서 기반 답변 생성 |
| `check_hallucination` | 조건부 엣지 | 환각 여부 판단 → END / generate 분기 |
