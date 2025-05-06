# LangGraph와 RAG 통합 구현

import os
from typing import TypedDict, List, Annotated, Dict, Any, Optional
import operator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_community import GoogleSearchResults
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
import json
import pandas as pd
import numpy as np

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CSE_ID_GENERAL = os.getenv("GOOGLE_CSE_ID_GENERAL")
GOOGLE_CSE_ID_SCHOLAR = os.getenv("GOOGLE_CSE_ID_SCHOLAR")

# 상태 정의 - RAG 관련 상태 추가
class AgentState(TypedDict):
    messages: Annotated[List[AIMessage | HumanMessage | ToolMessage | SystemMessage], operator.add]
    intermediate_steps: List[tuple]
    retrieved_documents: Optional[List[Document]]
    query_analysis: Optional[Dict[str, Any]]

# Google 검색 래퍼 생성
general_search_wrapper = GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_API_KEY,
    google_cse_id=GOOGLE_CSE_ID_GENERAL
)

scholar_search_wrapper = GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_API_KEY,
    google_cse_id=GOOGLE_CSE_ID_SCHOLAR
)

# 도구 정의: Google 일반 검색과 Google Scholar 검색
tools = [
    GoogleSearchResults(
        api_wrapper=general_search_wrapper,
        name="google_search",
        description="Search the web using Google Custom Search. Use this for general information and current events."
    ),
    GoogleSearchResults(
        api_wrapper=scholar_search_wrapper,
        name="google_scholar_search",
        description="Search academic papers using Google Scholar via Custom Search. Use this for scientific research and academic information."
    )
]

# LLM 초기화
model = ChatOpenAI(
    temperature=0,
    streaming=True,
    model="gpt-3.5-turbo-16k",
    openai_api_key=OPENAI_API_KEY
)
model_with_tools = model.bind_tools(tools)

# 문서 임베딩을 위한 설정
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 분석용 LLM 설정 (낮은 온도로 정확한 분석용)
analysis_model = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

# 번역 기능 추가
def translate_to_korean(text):
    """영어 텍스트를 한국어로 번역한다."""
    translator = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY
    )
    
    # 영어인지 확인 (간단한 휴리스틱)
    if any(ord(c) > 127 for c in text[:100]):  # 비ASCII 문자가 있으면 이미 한국어일 가능성 높음
        return text
    
    response = translator.invoke([
        HumanMessage(content=f"다음 텍스트를 한국어로 번역해주세요. 번역만 제공하고 다른 설명은 하지 마세요:\n\n{text}")
    ])
    
    return response.content

# 1. 쿼리 분석 노드: 사용자 질문 분석
def query_analysis_node(state: AgentState) -> dict:
    print("--- 쿼리 분석 노드 실행 ---")
    
    # 최신 사용자 메시지 추출
    user_message = next((msg.content for msg in reversed(state["messages"]) 
                         if isinstance(msg, HumanMessage)), "")
    
    # 쿼리 분석 프롬프트
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 사용자 질문을 분석하는 AI 전문가입니다. 
        다음 질문을 분석하여 JSON 형식으로 다음 정보를 제공하세요:
        1. query_type: 질문의 유형 (factual, analysis, opinion, technical, scholarly)
        2. domains: 관련된 지식 도메인 목록 (최대 3개)
        3. requires_rag: 내부 지식베이스 검색이 필요한지 (true/false)
        4. requires_web_search: 웹 검색이 필요한지 (true/false)
        5. requires_scholar_search: 학술 검색이 필요한지 (true/false)
        6. search_query: 검색에 적합한 질의어 (있는 경우)
        7. complexity: 질문의 복잡성 (simple, moderate, complex)
        """),
        ("user", f"다음 질문을 분석해주세요: {user_message}")
    ])
    
    # 분석 실행
    analysis_chain = analysis_prompt | analysis_model | StrOutputParser()
    analysis_result = analysis_chain.invoke({})
    
    # JSON 파싱 (오류 처리 포함)
    try:
        query_analysis = json.loads(analysis_result)
    except json.JSONDecodeError:
        # 파싱 오류 시 기본값 사용
        query_analysis = {
            "query_type": "factual",
            "domains": ["general"],
            "requires_rag": True,
            "requires_web_search": True,
            "requires_scholar_search": False,
            "search_query": user_message,
            "complexity": "moderate"
        }
    
    print(f"쿼리 분석 결과: {query_analysis}")
    
    return {
        "query_analysis": query_analysis
    }

# 라우터 노드: 질문 유형에 따라 처리 경로 결정
def router_node(state: AgentState) -> dict:
    print("--- 라우터 노드 실행 ---")
    query_analysis = state.get("query_analysis", {})
    query_type = query_analysis.get("query_type", "factual")
    # 현재는 모든 유형을 rag_retrieval로 라우팅, 확장 가능
    return {"next": "rag_retrieval"}

# 2. RAG 검색 노드: 벡터 저장소에서 관련 문서 검색
def rag_retrieval_node(state: AgentState) -> dict:
    print("--- RAG 검색 노드 실행 ---")
    
    # 기존 벡터 저장소 연결 또는 생성
    try:
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        print(f"기존 벡터 DB 연결 성공: 문서 수 = {vector_store._collection.count()}")
    except Exception as e:
        print(f"벡터 DB 연결 실패: {e}. 빈 RAG 결과 반환")
        return {"retrieved_documents": []}
    
    # 최신 사용자 메시지 추출
    user_message = next((msg.content for msg in reversed(state["messages"]) 
                         if isinstance(msg, HumanMessage)), "")
    
    # 쿼리 분석 결과 가져오기
    query_analysis = state.get("query_analysis", {})
    
    # RAG 검색이 필요한 경우만 실행
    if not query_analysis.get("requires_rag", True):
        print("쿼리 분석 결과 RAG가 필요하지 않음")
        return {"retrieved_documents": []}
    
    # 검색 실행
    try:
        search_query = query_analysis.get("search_query", user_message)
        docs = vector_store.similarity_search(search_query, k=4)
        print(f"RAG 검색 결과: {len(docs)}개 문서 찾음")
        
        # 각 문서의 메타데이터와 내용 출력
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "알 수 없는 출처")
            print(f"문서 {i+1}: {source}, 길이: {len(doc.page_content)} 글자")
        
        return {"retrieved_documents": docs}
    except Exception as e:
        print(f"RAG 검색 오류: {e}")
        return {"retrieved_documents": []}

# 3. Agent 노드: RAG 결과와 쿼리 분석을 포함하여 LLM 호출
def agent_node(state: AgentState) -> dict:
    print("--- Agent 노드 실행 ---")
    
    # 검색된 문서 정보 추출
    retrieved_docs = state.get("retrieved_documents", [])
    query_analysis = state.get("query_analysis", {})
    
    # 시스템 메시지 생성 (RAG 결과 포함)
    if retrieved_docs:
        rag_context = "\n\n".join([f"문서 {i+1} (출처: {doc.metadata.get('source', '알 수 없음')}):\n{doc.page_content}" 
                                for i, doc in enumerate(retrieved_docs)])
        
        system_message = SystemMessage(content=f"""
        다음은 당신의 지식 베이스에서 검색된 관련 정보입니다. 이 정보를 참고하여 사용자의 질문에 답변하세요.
        
        검색된 문서:
        {rag_context}
        
        위 문서 내용을 바탕으로 질문에 답변하되, 문서에 없는 내용은 자신의 지식을 활용하세요.
        내부 문서를 인용할 때는 '내부 문서 X에 따르면'과 같은 형식으로 언급하세요.
        """)
        
        # 시스템 메시지 추가
        current_messages = state["messages"]
        
        # 기존 메시지 목록에 시스템 메시지가 있는지 확인하고 업데이트 또는 추가
        system_message_exists = False
        updated_messages = []
        
        for msg in current_messages:
            if isinstance(msg, SystemMessage):
                updated_messages.append(system_message)  # 기존 시스템 메시지 대체
                system_message_exists = True
            else:
                updated_messages.append(msg)
        
        if not system_message_exists:
            # 시스템 메시지가 없으면 맨 앞에 추가
            updated_messages = [system_message] + updated_messages
    else:
        # RAG 결과가 없는 경우 기존 메시지 사용
        updated_messages = state["messages"]
    
    # 쿼리 유형에 따라 웹 검색 도구 사용 여부 결정
    if query_analysis.get("requires_web_search", False) or query_analysis.get("requires_scholar_search", False):
        response = model_with_tools.invoke(updated_messages)
    else:
        # 웹 검색이 필요 없는 경우 도구 없이 호출
        response = model.invoke(updated_messages)
    
    return {"messages": [response]}

# 4. Tool 노드: Google 검색 또는 Google Scholar 검색 실행
def tool_node(state: AgentState) -> dict:
    print("--- Tool 노드 실행 ---")
    last_message = state["messages"][-1]
    tool_results = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call.get("id", "unknown")
        try:
            if tool_name in ["google_search", "google_scholar_search"]:
                # 도구 선택
                tool = next(t for t in tools if t.name == tool_name)
                result = tool.invoke(tool_args["query"])
                tool_results.append((tool_id, tool_name, result))
            else:
                raise ValueError(f"지원되지 않는 도구: {tool_name}")
        except Exception as e:
            print(f"도구 실행 오류: {e}")
            tool_results.append((tool_id, tool_name, f"오류: {e}"))
    
    tool_messages = [
        ToolMessage(
            content=str(result),
            tool_call_id=tool_id,
            name=tool_name
        ) for tool_id, tool_name, result in tool_results
    ]
    return {
        "messages": tool_messages,
        "intermediate_steps": state["intermediate_steps"] + [(tool_name, result) for tool_id, tool_name, result in tool_results]
    }

# 5. 문서 추가 노드: 새 문서를 벡터 DB에 추가
def add_document_node(state: AgentState, documents: List[Document]) -> dict:
    print("--- 문서 추가 노드 실행 ---")
    
    if not documents:
        print("추가할 문서가 없습니다.")
        return {}
    
    # 텍스트 분할기 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    
    # 문서 분할
    split_docs = text_splitter.split_documents(documents)
    print(f"문서 {len(documents)}개를 {len(split_docs)}개 청크로 분할")
    
    # 벡터 저장소 연결 또는 생성
    try:
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        # 문서 추가
        vector_store.add_documents(split_docs)
        vector_store.persist()  # 변경사항 저장
        print(f"벡터 DB에 문서 추가 완료. 현재 문서 수: {vector_store._collection.count()}")
        
    except Exception as e:
        print(f"벡터 DB 작업 중 오류 발생: {e}")
    
    return {}

# 조건부 엣지: 도구 호출 여부 결정
def should_continue(state: AgentState) -> str:
    print("--- 조건부 엣지 평가 ---")
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("결정: 도구 사용 필요")
        return "tool_node"
    print("결정: 최종 응답 (종료)")
    return END

# 파일에서 문서 로드 함수
def load_documents_from_file(file_path, source_name=None):
    """파일에서 문서를 로드한다."""
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없음: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 메타데이터 설정 (파일명이 없으면 경로 사용)
        metadata = {"source": source_name or file_path}
        
        # Document 객체 생성
        doc = Document(page_content=content, metadata=metadata)
        return [doc]
    
    except Exception as e:
        print(f"파일 로드 오류: {e}")
        return []

# 디렉토리에서 모든 문서 로드 함수
def load_documents_from_directory(directory_path):
    """디렉토리에서 모든 텍스트 파일을 로드한다."""
    documents = []
    if not os.path.exists(directory_path):
        print(f"디렉토리를 찾을 수 없음: {directory_path}")
        return documents
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt') or filename.endswith('.md'):
            file_path = os.path.join(directory_path, filename)
            docs = load_documents_from_file(file_path, filename)
            documents.extend(docs)
    
    print(f"{directory_path}에서 {len(documents)}개 문서 로드됨")
    return documents

# 그래프 정의 및 컴파일
def build_rag_agent_graph():
    # 그래프 빌더 초기화
    graph_builder = StateGraph(AgentState)
    
    # 초기 상태 설정
    graph_builder.set_initial_state({
        "messages": [],
        "intermediate_steps": [],
        "retrieved_documents": None,
        "query_analysis": None
    })
    
    # 노드 추가
    graph_builder.add_node("query_analysis", query_analysis_node)
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("rag_retrieval", rag_retrieval_node)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tool", tool_node)
    
    # 엣지 추가 (처리 흐름 정의)
    graph_builder.set_entry_point("query_analysis")
    graph_builder.add_edge("query_analysis", "router")
    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {"rag_retrieval": "rag_retrieval"}
    )
    graph_builder.add_edge("rag_retrieval", "agent")
    
    # 조건부 엣지 추가
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tool_node": "tool", END: END}
    )
    graph_builder.add_edge("tool", "agent")
    
    # 그래프 컴파일
    return graph_builder.compile()

# 메인 실행 코드
if __name__ == "__main__":
    print("\n=== LangGraph + RAG 에이전트 ===")
    print("내부 지식베이스와 Google 검색을 결합하여 질문에 답변합니다.")
    
    # 초기 설정 (벡터 DB 준비)
    print("\n--- 초기 설정 ---")
    
    # 벡터 DB 디렉토리 확인 및 생성
    if not os.path.exists("./chroma_db"):
        os.makedirs("./chroma_db")
        print("벡터 DB 디렉토리 생성됨")
    
    # 샘플 문서 디렉토리 확인 및 생성
    if not os.path.exists("./documents"):
        os.makedirs("./documents")
        print("문서 디렉토리 생성됨. 여기에 텍스트 파일을 추가하세요.")
    
    # 그래프 빌드
    graph = build_rag_agent_graph()
    
    # 문서 로드 (있다면)
    docs = load_documents_from_directory("./documents")
    if docs:
        # 벡터 저장소 준비 (처음 실행 시 문서 추가)
        try:
            vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            current_count = vector_store._collection.count()
            
            if current_count == 0:
                print("벡터 DB가 비어 있습니다. 문서를 추가합니다...")
                add_document_node({}, docs)
            else:
                print(f"기존 벡터 DB 발견: {current_count}개 문서 포함")
        except Exception as e:
            print(f"벡터 DB 초기화 오류: {e}")
            print("새 벡터 DB 생성 및 문서 추가...")
            add_document_node({}, docs)
    
    # 메인 루프
    while True:
        user_query = input("\n질문을 입력하세요 (종료하려면 'exit' 입력, 문서 추가는 'add:파일경로'): ")
        
        if user_query.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break
        
        # 문서 추가 명령 처리
        if user_query.startswith('add:'):
            file_path = user_query[4:].strip()
            if os.path.exists(file_path):
                docs = load_documents_from_file(file_path)
                if docs:
                    add_document_node({}, docs)
                    print(f"문서 '{file_path}'가 지식 베이스에 추가되었습니다.")
                else:
                    print("문서 로드 실패")
            else:
                print(f"파일을 찾을 수 없음: {file_path}")
            continue
        
        # 일반 질문 처리
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "intermediate_steps": [],
            "retrieved_documents": None,
            "query_analysis": None
        }
        
        print("\n--- 그래프 실행 시작 ---")
        for event in graph.stream(initial_state, {"recursion_limit": 10}):
            event_node = list(event.keys())[0]
            event_data = event[event_node]
            print(f"\n--- {event_node} 실행 결과 ---")
            if "messages" in event_data:
                for msg in event_data["messages"]:
                    if isinstance(msg, AIMessage):
                        # AI 응답을 한국어로 번역
                        korean_content = translate_to_korean(msg.content)
                        print("\n[AI 응답]:", korean_content)
                    elif isinstance(msg, ToolMessage):
                        # 도구 결과는 번역하지 않음 (검색 결과 등은 원문 유지)
                        print(f"\n[도구 결과 ({msg.name})]:", msg.content[:200], "..." if len(msg.content) > 200 else "")
            
        print("\n=== 응답 완료 ===")