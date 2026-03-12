# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# %%
# 벡터DB 불러오기
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os

index_name = 'tax-index'
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding
)

# %%
#query = "연봉 5천만원 거주자의 소득세는 얼마인가요"

retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
#retriever.invoke(query)

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

# %%
def retrieve(state: AgentState) -> AgentState:
    """ 사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다. """
    
    query = state['query']
    docs = retriever.invoke(query)
    
    return {'context': docs}
    

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-5.2')

# %%
from langchain import hub

generate_prompt = hub.pull("rlm/rag-prompt")

# %%
def generate(state: AgentState) -> AgentState:
    """ 주어진 state를 기반으로 RAG 체인을 사용하여 응답을 생성합니다. """
    
    context = state['context']
    query = state['query']
    
    rag_chain = generate_prompt | llm
    response = rag_chain.invoke({'question': query, 'context': context})
    
    return {'answer': response}

# %%
from langchain_core.prompts import ChatPromptTemplate
#doc_relevence_prompt = hub.pull('langchain-ai/rag-document-relevance')
doc_relevence_prompt = ChatPromptTemplate.from_template(f"""
     당신은 검색된 문서가 사용자의 질문과 관련이 있는지 판별하는 전문가 이다.
     해당 질문(question)에 관련 문서(documents)이면 1을 리턴하고 그렇치 않으면 0을 리턴한다.
     
     question
     {{question}}
     
     documents
     {{documents}}
""")

# %%
from typing import Literal

def check_doc_relevence(state: AgentState) -> Literal['generate', 'rewrite']:
    """ 주어진 state를 기반으로 문서의 관련성을 판단합니다. """
    
    context = state['context']
    query = state['query']
    
    doc_relevence_chain = doc_relevence_prompt | llm
    response = doc_relevence_chain.invoke({'question': query, 'documents': context})    
    print(f'relevance: {response}')
    
    if response.content == '1':
        return 'generate'
    return 'rewrite'



# %%
#query = '연봉 5천만원 거주자의 소득세는 얼마인가요?'
#query = '연봉 5천만원 세금 얼마인가요?'

# context = retriever.invoke(query)
# relevence_state = {'query': query, 'context': context}
# result = check_doc_relevence(relevence_state)
# result

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

dictionary = ["사람을 나타내는 표현 -> 거주자"]

rewrite_prompt = ChatPromptTemplate.from_template(f"""
     사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해 주세요.
     만약 변경할 필요가 없다고 판단 된다면, 사용자의 질문을 변경하지 않아도 됩니다.
     질문내용 중에 '*' 표시를 제거 한다.
     
     사전 : {dictionary}                                      
     
     질문 : {{question}}
""")

# %%
def rewrite(state: AgentState) -> AgentState:
    """ 사용자의 질문을 사전을 고려하여 질문을 변경합니다."""
    
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    response = rewrite_chain.invoke({'question': query})
    print(f'rewrite: {response}')
    return {'query': response}

# %%
from langchain_core.prompts import PromptTemplate

hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}
student_answer: {student_answer}
""")

# %%
def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']
    context = state['context']
    
    hallucitation_chain = hallucination_prompt | llm | StrOutputParser()
    response = hallucitation_chain.invoke({'student_answer': answer, 'documents': context})
    
    print('거짓말 유무: ', response)
    
    return response

# %%
#query = '연봉 5천만원 거주자의 소득세는 얼마인가요?'
# query = '연봉 5천만원 세금 얼마인가요?'

# context = retriever.invoke(query)
# generate_state = {'query': query, 'context': context}
# answer = generate(generate_state)
# print(f'answer: {answer}')

# hallucination_state = {'answer': answer, 'context': context}
# check_hallucination(hallucination_state)


# %%
builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("rewrite", rewrite)

# %%
from langgraph.graph import START, END

builder.add_edge(START, 'retrieve')
builder.add_conditional_edges(
    'retrieve',
    check_doc_relevence,
    {
       'generate': 'generate',
       'rewrite': 'rewrite'
    }
)
builder.add_edge('rewrite', 'retrieve')
builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
       'not hallucinated': END,
       'hallucinated': 'generate'
    }
)

# %%
graph = builder.compile()

# %%
# from IPython.display import Image, display

# display(Image(graph.get_graph().draw_mermaid_png()))

# %%
#query = '연봉 5천만원 거주자의 소득세는 얼마인가요?'
#연봉 5천만원 직장인의 소득세는?
# initial_state = {'query': "연봉 5천만원 직장인의 소득세는 얼마인가요?"}
# response = graph.invoke(initial_state)
# response

# %%



