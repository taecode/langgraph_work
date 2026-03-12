# 03_langgraph.ipynb 단계별 개념 설명

> 레스토랑 AI 어시스턴트를 만들면서 LangGraph의 핵심 개념을 학습하는 노트북입니다.
> "메뉴 DB에서 찾고 → 없으면 웹에서 찾고 → AI가 답변하는" 에이전트를 단계적으로 구현합니다.

---

## 전체 흐름 한눈에 보기

```
사용자 질문
    │
    ▼
[AI 판단] ──── 메뉴 관련? ──→ search_menu (벡터 DB 검색)
    │                                │
    │         최신/외부 정보? ──→ search_web (인터넷 검색)
    │                                │
    │         알고 있는 것?  ──→ 도구 없이 직접 답변
    │                                │
    └────────────────────────────────┘
    ▼
최종 답변 생성
```

---

## STEP 1 — 환경 설정

```python
from dotenv import load_dotenv
load_dotenv()
```

### 개념

`.env` 파일에 저장된 API Key(OpenAI, Tavily)를 불러옵니다.
코드에 키를 직접 쓰지 않고 외부 파일로 분리해서 보안을 지킵니다.

```
# .env 파일 예시
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

---

## STEP 2 — 벡터 DB 준비

```python
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

menu_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_menu",
    persist_directory="./chroma_db",
)
```

### 개념: Embedding이란?

텍스트를 숫자 배열(벡터)로 변환해서 **의미가 비슷한 문장을 찾는** 기술입니다.

```
"스테이크"    → [0.12, -0.87, 0.34, ...]
"소고기 요리" → [0.11, -0.85, 0.36, ...]  ← 의미가 비슷 → 숫자도 비슷
"아이스크림"  → [0.91,  0.23,-0.44, ...]  ← 의미가 다름 → 숫자도 다름
```

### 개념: Chroma란?

이 숫자 벡터들을 저장해두는 **로컬 벡터 데이터베이스**입니다.
`./chroma_db` 폴더에 영구 저장되어 다음 실행 시에도 재사용 가능합니다.

---

## STEP 3 — 도구(Tool) 2개 정의

### 도구 1: `search_menu` — 메뉴 DB 검색

```python
@tool
def search_menu(query: str) -> List[str]:
    """레스토랑 메뉴에서 정보를 검색합니다."""
    docs = menu_db.similarity_search(query, k=2)  # 유사한 문서 2개 찾기
    ...
```

#### 개념: RAG (Retrieval-Augmented Generation)

AI가 학습하지 않은 **내부 데이터**를 실시간으로 찾아서 답변에 활용하는 방식입니다.

```
사용자 질문
    │
    ▼
벡터 DB에서 관련 문서 검색 (similarity_search)
    │
    ▼
AI에게 [문서 내용 + 질문] 함께 전달
    │
    ▼
정확한 답변 생성
```

> 🔑 핵심: AI의 학습 데이터에 없는 내용도 답할 수 있게 됩니다.

---

### 도구 2: `search_web` — 인터넷 검색

```python
@tool
def search_web(query: str) -> str:
    """최신 정보를 인터넷으로 검색합니다."""
    tavily_search = TavilySearchResults(max_results=3)
    docs = tavily_search.invoke(query)
    ...
```

#### 개념: Tavily란?

AI 에이전트용으로 최적화된 검색 API입니다.
메뉴 DB에 없는 정보나 **최신 정보**를 웹에서 실시간으로 가져옵니다.

---

## STEP 4 — LLM에 도구 연결 (Tool Binding)

```python
llm = ChatOpenAI(model="gpt-5-mini")
tools = [search_menu, search_web]
llm_with_tools = llm.bind_tools(tools=tools)
```

### 개념: Tool Binding이란?

도구를 LLM에 묶어주면, AI가 질문을 보고 **스스로 어떤 도구를 쓸지 판단**합니다.

| 질문                    | AI의 판단            | 이유                |
| ----------------------- | -------------------- | ------------------- |
| `"스테이크 가격은?"`  | `search_menu` 사용 | 내부 DB에 있는 정보 |
| `"LangGraph가 뭐야?"` | `search_web` 사용  | 최신/외부 정보 필요 |
| `"1+2는 얼마?"`       | 도구 없이 직접 답변  | 계산은 도구 불필요  |

---

## STEP 5 — AI의 도구 선택 확인

```python
tool_call = llm_with_tools.invoke([HumanMessage("스테이크 메뉴의 가격은 얼마인가요?")])
print(tool_call.additional_kwargs)
# 출력: {'tool_calls': [{'name': 'search_menu', 'arguments': '{"query":"스테이크 메뉴 가격"}', ...}]}
```

### 주의: 이 단계는 "선택"만 합니다

```
llm_with_tools.invoke() 호출
        │
        ▼
AI: "search_menu를 써야겠다" ← 결정만 함
        │
   실제 실행은 아직 안 함!
        │
        ▼
tool_calls 정보가 반환됨 (함수명, 인자값)
```

---

## STEP 6 — ToolNode: 도구 실제 실행

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools=tools)
result = tool_node.invoke({"messages": [tool_call]})
```

### 개념: ToolNode란?

AI가 "search_menu를 써야겠다"고 결정한 내용을 보고, **실제 함수를 실행**해서 결과를 돌려주는 실행자입니다.

```
AI 결정 (tool_calls 정보)
        │
        ▼
   ToolNode
        │
        ▼
실제 search_menu() 함수 호출
        │
        ▼
결과: "시그니처 스테이크 ₩35,000..."
```

---

## STEP 7 — 간단한 에이전트 (create_react_agent)

```python
from langgraph.prebuilt import create_react_agent

graph = create_react_agent(model=llm, tools=tools)
result = graph.invoke({"messages": [HumanMessage("스테이크 메뉴의 가격은 얼마인가요?")]})
```

### 개념: ReAct 패턴이란?

AI가 **생각하고 → 행동하고 → 결과를 보고 → 다시 생각하는** 루프 패턴입니다.

```
① Reason  → "스테이크 가격을 알려면 메뉴 DB를 찾아야겠다"
      │
      ▼
② Act     → search_menu("스테이크") 실행
      │
      ▼
③ Observe → "시그니처 스테이크 ₩35,000 발견"
      │
      ▼
④ Reason  → "이걸로 답변할 수 있다. 도구 더 필요 없음"
      │
      ▼
⑤ 최종 답변 → "스테이크 가격은 ₩35,000입니다"
```

`create_react_agent`는 이 루프를 **자동으로 만들어주는 편의 함수**입니다.

### 생성된 그래프 구조

```
START
  │
  ▼
[agent] ── tool_calls 있음? ── YES ──→ [tools]
  ▲                                       │
  └───────────────────────────────────────┘
  │
  └── tool_calls 없음? ──→ END
```

---

## STEP 8 — 시스템 프롬프트 정의

```python
system_prompt = """
You are an AI assistant...

When using tools, cite the source as follows:
[Source: tool_name | document_title | url/file_path]
...
"""
```

### 개념: 시스템 프롬프트란?

AI의 **행동 지침서**입니다. 대화가 시작되기 전에 AI에게 역할, 말투, 출력 형식 등을 사전에 지시합니다.

```
시스템 프롬프트 → "너는 레스토랑 AI야. 출처를 반드시 표기해"
HumanMessage   → "스테이크 가격은?"
AIMessage      → "[Source: search_menu | ...] 가격은 ₩35,000입니다"
```

> `create_react_agent`는 커스터마이징이 제한적이기 때문에,
> 다음 단계에서 **직접 그래프를 만들어** 더 세밀하게 제어합니다.

---

## STEP 9 — 직접 그래프 만들기 (핵심)

이 단계가 LangGraph의 **핵심**입니다.

### 9-1. State (상태) 정의

```python
class GraphState(MessagesState):
    pass
```

State는 그래프를 흐르는 **공유 데이터 저장소**입니다.
모든 노드가 이 State를 읽고, 쓰면서 데이터를 주고받습니다.

```
State = {
    "messages": [
        HumanMessage("스테이크 가격?"),      ← 사용자 입력
        AIMessage("search_menu 호출할게요"), ← call_mode 노드가 추가
        ToolMessage("₩35,000"),             ← execute_tools 노드가 추가
        AIMessage("₩35,000입니다")          ← call_mode 노드가 추가
    ]
}
```

---

### 9-2. 노드(Node) 정의

```python
def call_mode(state: GraphState) -> GraphState:
    system_message = SystemMessage(content=system_prompt)
    messages = [system_message] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

노드는 **State를 받아서 작업하고, 새로운 State를 돌려주는 함수**입니다.

```
State (입력)
    │
    ▼
  노드 함수 실행
    │
    ▼
State (출력) ← messages에 새 내용 추가됨
```

---

### 9-3. 조건부 엣지(Conditional Edge) 정의

```python
def should_continue(state: GraphState):
    last_message = state['messages'][-1]

    if last_message.tool_calls:   # AI가 도구를 쓰겠다고 했으면
        return "execute_tools"    # → ToolNode로 이동

    return END                    # 도구 불필요 → 종료
```

| 엣지 종류             | 설명                                             |
| --------------------- | ------------------------------------------------ |
| **일반 엣지**   | 항상 같은 곳으로 이동 (`A → B`)               |
| **조건부 엣지** | 상황에 따라 다른 곳으로 분기 (`A → B 또는 C`) |

---

### 9-4. 그래프 조립

```python
builder = StateGraph(GraphState)

# 노드 등록
builder.add_node("call_mode", call_mode)
builder.add_node("execute_tools", ToolNode(tools))

# 엣지 연결
builder.add_edge(START, "call_mode")            # 시작 → AI 판단
builder.add_conditional_edges(                   # AI 판단 후 분기
    "call_mode",
    should_continue,
    {
        "execute_tools": "execute_tools",        # 도구 필요 → 실행
        END: END                                  # 불필요 → 종료
    }
)
builder.add_edge("execute_tools", "call_mode")  # 실행 후 → 다시 AI 판단

graph = builder.compile()
```

### 완성된 그래프 구조

```
START
  │
  ▼
[call_mode] ── tool_calls 있음? ── YES ──→ [execute_tools]
  ▲                                               │
  │                                               │
  └───────────────────────────────────────────────┘
  │
  └── tool_calls 없음? ──→ END
```

이 루프가 반복되면서 AI가 "더 이상 도구가 필요 없다"고 판단할 때까지 계속 실행됩니다.

---

## STEP 10 — 최종 실행 및 결과 확인

```python
inputs = {
    "messages": [HumanMessage("스테이크 메뉴의 가격은 얼마인가요?")]
}

result = graph.invoke(inputs)

for message in result.get("messages", []):
    message.pretty_print()
```

### 실제 실행 흐름

```
① 사용자: "스테이크 가격은?"
        │
        ▼
② [call_mode] AI 판단
   → tool_calls: [search_menu(query="스테이크")]
        │
        ▼  (tool_calls 있음)
③ [execute_tools] search_menu 실행
   → "시그니처 스테이크 ₩35,000, 안심 스테이크 샐러드 ₩26,000"
        │
        ▼
④ [call_mode] AI 다시 판단
   → tool_calls: [] (더 이상 도구 불필요)
        │
        ▼  (tool_calls 없음)
⑤ END → "시그니처 스테이크의 가격은 ₩35,000입니다
         [Source: search_menu | 시그니처 스테이크 | ./data/restaurant_menu.txt]"
```

---

## 핵심 개념 총정리

| 용어                       | 한 줄 설명                                             |
| -------------------------- | ------------------------------------------------------ |
| **Embedding**        | 텍스트를 숫자 벡터로 변환 — 의미 기반 검색의 핵심     |
| **Chroma**           | 벡터를 저장하는 로컬 데이터베이스                      |
| **RAG**              | DB에서 관련 문서를 찾아 AI 답변 품질을 높이는 기법     |
| **Tool**             | AI가 호출할 수 있는 외부 함수 (`@tool` 데코레이터)   |
| **Tool Binding**     | LLM이 어떤 도구를 쓸 수 있는지 알려주는 것             |
| **ToolNode**         | AI가 선택한 도구를 실제로 실행하는 노드                |
| **ReAct**            | 생각(Reason) → 행동(Act) → 관찰(Observe) 반복 패턴   |
| **State**            | 그래프 전체가 공유하는 데이터 저장소 (메시지 히스토리) |
| **Node**             | 그래프에서 작업을 수행하는 함수                        |
| **Edge**             | 노드와 노드를 연결하는 흐름                            |
| **Conditional Edge** | 조건에 따라 다른 노드로 분기하는 엣지                  |
| **StateGraph**       | 노드와 엣지를 조립해 그래프를 만드는 빌더              |

---

## create_react_agent vs 직접 구현 비교

| 항목                      | `create_react_agent` | 직접 구현 (`StateGraph`) |
| ------------------------- | ---------------------- | -------------------------- |
| **코드량**          | 적음 (3줄)             | 많음 (30줄+)               |
| **커스터마이징**    | 제한적                 | 자유로움                   |
| **시스템 프롬프트** | 기본 제공              | 직접 삽입 가능             |
| **노드 추가**       | 불가                   | 자유롭게 추가              |
| **학습 목적**       | 빠른 프로토타이핑      | 내부 구조 이해 필수        |

> 실무에서는 간단한 경우 `create_react_agent`, 복잡한 워크플로우는 `StateGraph` 직접 구현을 사용합니다.
