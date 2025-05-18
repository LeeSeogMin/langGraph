# app.py
# LangGraph+RAG 에이전트를 위한 Streamlit 웹 인터페이스

import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 문서 관리 모듈 임포트
from document_manager import DocumentManager

# LangGraph 에이전트 모듈 임포트
from rag_agent import (
    build_rag_agent_graph,
    translate_to_korean,
    AgentState,
    HumanMessage
)

# 환경 변수 로드
load_dotenv()

# API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID_GENERAL = os.getenv("GOOGLE_CSE_ID_GENERAL", "")
GOOGLE_CSE_ID_SCHOLAR = os.getenv("GOOGLE_CSE_ID_SCHOLAR", "")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit 페이지 설정
st.set_page_config(
    page_title="LangGraph+RAG 에이전트",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_manager" not in st.session_state:
    st.session_state.document_manager = None
    
if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = None

if "api_keys_valid" not in st.session_state:
    st.session_state.api_keys_valid = False

def initialize_managers():
    """문서 관리자와 에이전트 그래프 초기화"""
    if not st.session_state.api_keys_valid:
        st.warning("API 키가 설정되지 않았습니다. 설정 탭에서 API 키를 입력하세요.")
        return False
    
    try:
        # 문서 관리자 초기화
        if st.session_state.document_manager is None:
            st.session_state.document_manager = DocumentManager(
                vector_db_path="./chroma_db",
                openai_api_key=OPENAI_API_KEY
            )
            logger.info("문서 관리자 초기화 완료")
        
        # 에이전트 그래프 초기화
        if st.session_state.agent_graph is None:
            st.session_state.agent_graph = build_rag_agent_graph()
            logger.info("에이전트 그래프 초기화 완료")
        
        return True
    
    except Exception as e:
        logger.error(f"초기화 오류: {e}")
        st.error(f"초기화 중 오류가 발생했습니다: {e}")
        return False

def validate_api_keys():
    """API 키 유효성 검사"""
    if not OPENAI_API_KEY:
        st.sidebar.error("OpenAI API 키가 설정되지 않았습니다.")
        return False
    
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID_GENERAL:
        st.sidebar.warning("Google 검색 API 키가 설정되지 않았습니다. 웹 검색 기능을 사용할 수 없습니다.")
    
    return True if OPENAI_API_KEY else False

def send_message(query: str):
    """에이전트에 메시지 전송 및 응답 처리"""
    if not query.strip():
        return
    
    if not initialize_managers():
        return
    
    # 사용자 메시지 추가
    user_message = {"role": "user", "content": query, "timestamp": datetime.now().strftime("%H:%M:%S")}
    st.session_state.chat_history.append(user_message)
    
    # 처리 중 메시지
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 처리 중...")
    
    try:
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "intermediate_steps": [],
            "retrieved_documents": [],  # None 대신 빈 리스트로 초기화
            "query_analysis": {}  # None 대신 빈 딕셔너리로 초기화
        }
        
        # 에이전트 그래프 실행
        last_ai_response = ""
        search_results = []
        retrieved_docs = []
        
        for event in st.session_state.agent_graph.stream(initial_state, {"recursion_limit": 10}):
            event_node = list(event.keys())[0]
            event_data = event[event_node]
            
            if event_node == "rag_retrieval" and "retrieved_documents" in event_data:
                retrieved_docs = event_data["retrieved_documents"] or []
                
                # 검색된 문서 정보 표시
                if retrieved_docs:
                    doc_info = []
                    for i, doc in enumerate(retrieved_docs):
                        source = doc.metadata.get("source", "알 수 없는 출처")
                        doc_info.append(f"📄 문서 {i+1}: {source} (관련도: {100 - i*10:.0f}%)")
                    
                    message_placeholder.markdown(
                        "🔍 관련 문서를 찾았습니다:\n" + "\n".join(doc_info) + "\n\n🧠 답변 생성 중..."
                    )
            
            if "messages" in event_data:
                for msg in event_data["messages"]:
                    # 도구 메시지 처리 (검색 결과)
                    if hasattr(msg, 'name') and msg.name in ["google_search", "google_scholar_search"]:
                        search_results.append({
                            "tool": msg.name,
                            "content": msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                        })
                        tool_name = "Google 검색" if msg.name == "google_search" else "Google Scholar 검색"
                        message_placeholder.markdown(
                            f"🔍 {tool_name} 결과를 가져왔습니다. 답변 생성 중..."
                        )
                    
                    # AI 응답 처리
                    elif hasattr(msg, 'content') and not hasattr(msg, 'name'):
                        korean_content = translate_to_korean(msg.content)
                        last_ai_response = korean_content
                        message_placeholder.markdown(korean_content)
        
        # 최종 AI 응답 추가
        if last_ai_response:
            # 검색된 문서 및 검색 결과 추가
            metadata = {
                "retrieved_docs": [
                    {"source": doc.metadata.get("source", "알 수 없는 출처")} 
                    for doc in retrieved_docs
                ],
                "search_results": search_results
            }
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": last_ai_response, 
                "metadata": metadata,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        
    except Exception as e:
        logger.error(f"에이전트 실행 오류: {e}")
        message_placeholder.markdown(f"😵 오류가 발생했습니다: {e}")
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"오류가 발생했습니다: {e}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

def display_chat_history():
    """채팅 기록 표시"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 메타데이터 표시 (확장 가능)
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                
                # 검색된 문서 표시
                if metadata.get("retrieved_docs"):
                    with st.expander("📚 참조된 내부 문서", expanded=False):
                        for i, doc in enumerate(metadata["retrieved_docs"]):
                            st.markdown(f"{i+1}. **출처**: {doc['source']}")
                
                # 검색 결과 표시
                if metadata.get("search_results"):
                    with st.expander("🔍 웹 검색 결과", expanded=False):
                        for result in metadata["search_results"]:
                            tool_name = "Google 검색" if result["tool"] == "google_search" else "Google Scholar 검색"
                            st.markdown(f"**{tool_name}**:")
                            st.text(result["content"])

def display_document_management():
    """문서 관리 인터페이스"""
    if not initialize_managers():
        return
    
    document_manager = st.session_state.document_manager
    
    st.markdown("## 📚 문서 관리")
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📄 문서 추가", "📋 문서 목록", "📊 통계"])
    
    # 문서 추가 탭
    with tab1:
        st.markdown("### 새 문서 추가")
        
        upload_method = st.radio(
            "추가 방식 선택:",
            ["파일 업로드", "디렉토리 경로 지정"]
        )
        
        tags = st.text_input("태그 (쉼표로 구분):", "").strip()
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        
        if upload_method == "파일 업로드":
            uploaded_files = st.file_uploader(
                "문서 파일 선택 (txt, pdf, md, csv 등)",
                accept_multiple_files=True,
                type=["txt", "pdf", "md", "csv", "xlsx"]
            )
            
            if uploaded_files and st.button("선택한 파일 추가", type="primary"):
                for uploaded_file in uploaded_files:
                    # 임시 파일 저장
                    temp_file_path = os.path.join("./temp_uploads", uploaded_file.name)
                    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                    
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 문서 추가
                    success = document_manager.add_document(temp_file_path, tag_list)
                    
                    # 임시 파일 삭제
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    
                    if success:
                        st.success(f"'{uploaded_file.name}' 추가 완료")
                    else:
                        st.error(f"'{uploaded_file.name}' 추가 실패")
        
        else:  # 디렉토리 경로 지정
            directory_path = st.text_input("문서 디렉토리 경로:", "./documents")
            
            if directory_path and st.button("디렉토리 내 문서 추가", type="primary"):
                if not os.path.exists(directory_path):
                    st.error(f"디렉토리를 찾을 수 없습니다: {directory_path}")
                else:
                    with st.spinner("문서 추가 중..."):
                        results = document_manager.add_directory(directory_path, tag_list)
                        st.success(
                            f"디렉토리 처리 완료: {results['success']}개 성공, "
                            f"{results['failed']}개 실패, {results['skipped']}개 건너뜀"
                        )
    
    # 문서 목록 탭
    with tab2:
        st.markdown("### 인덱싱된 문서 목록")
        
        # 문서 목록 가져오기
        docs_df = document_manager.list_documents()
        
        if docs_df.empty:
            st.info("인덱싱된 문서가 없습니다. '문서 추가' 탭에서 문서를 추가하세요.")
        else:
            st.dataframe(docs_df, use_container_width=True)
            
            # 문서 삭제 기능
            if not docs_df.empty:
                st.markdown("### 문서 삭제")
                doc_to_remove = st.selectbox(
                    "삭제할 문서 선택:",
                    options=docs_df["filename"].tolist()
                )
                
                if st.button("선택한 문서 삭제", type="secondary"):
                    if document_manager.remove_document(filename=doc_to_remove):
                        st.success(f"'{doc_to_remove}' 삭제 완료")
                    else:
                        st.error(f"'{doc_to_remove}' 삭제 실패")
    
    # 통계 탭
    with tab3:
        st.markdown("### 문서 통계")
        
        stats = document_manager.get_statistics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 기본 통계
            st.metric("총 문서 수", stats["total_documents"])
            st.metric("총 청크 수", stats["total_chunks"])
            st.markdown(f"**마지막 업데이트**: {stats['last_updated']}")
            
            # 태그 통계
            if stats["tags"]:
                st.markdown("#### 태그별 문서 수")
                tags_df = pd.DataFrame(
                    {"태그": list(stats["tags"].keys()), "문서 수": list(stats["tags"].values())}
                ).sort_values("문서 수", ascending=False)
                
                st.dataframe(tags_df, use_container_width=True)
        
        with col2:
            # 파일 유형 통계
            if stats["file_types"]:
                st.markdown("#### 파일 유형별 문서 수")
                
                fig = px.pie(
                    names=list(stats["file_types"].keys()),
                    values=list(stats["file_types"].values()),
                    title="파일 유형 분포"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 청크 대비 문서 비율
            if stats["total_documents"] > 0:
                st.markdown("#### 평균 청크 수")
                avg_chunks = stats["total_chunks"] / stats["total_documents"]
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_chunks,
                    title={"text": "문서당 평균 청크 수"},
                    gauge={"axis": {"range": [0, max(10, avg_chunks * 1.5)]}}
                ))
                st.plotly_chart(fig, use_container_width=True)

def display_settings():
    """설정 인터페이스"""
    st.markdown("## ⚙️ 설정")
    
    # API 키 입력
    st.markdown("### API 키 설정")
    
    openai_key = st.text_input(
        "OpenAI API 키",
        value=OPENAI_API_KEY,
        type="password",
        help="OpenAI API 키를 입력하세요. 환경 변수에서 로드된 값이 있으면 자동으로 채워집니다."
    )
    
    google_key = st.text_input(
        "Google API 키",
        value=GOOGLE_API_KEY,
        type="password",
        help="Google Custom Search API 키를 입력하세요."
    )
    
    google_cse_id = st.text_input(
        "Google CSE ID (일반)",
        value=GOOGLE_CSE_ID_GENERAL,
        help="Google Custom Search Engine ID를 입력하세요."
    )
    
    google_cse_id_scholar = st.text_input(
        "Google CSE ID (Scholar)",
        value=GOOGLE_CSE_ID_SCHOLAR,
        help="학술 검색용 Google Custom Search Engine ID를 입력하세요."
    )
    
    # API 키 저장
    if st.button("API 키 저장", type="primary"):
        # .env 파일 업데이트
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY={openai_key}\n")
            f.write(f"GOOGLE_API_KEY={google_key}\n")
            f.write(f"GOOGLE_CSE_ID_GENERAL={google_cse_id}\n")
            f.write(f"GOOGLE_CSE_ID_SCHOLAR={google_cse_id_scholar}\n")
        
        st.success("API 키가 저장되었습니다. 애플리케이션을 재시작하세요.")
    
    # 설정 정보
    st.markdown("### 벡터 DB 설정")
    vector_db_path = "./chroma_db"
    
    if os.path.exists(vector_db_path):
        st.info(f"벡터 DB 경로: {os.path.abspath(vector_db_path)}")
        db_size = sum(os.path.getsize(os.path.join(root, file)) for root, _, files in os.walk(vector_db_path) for file in files)
        st.metric("벡터 DB 크기", f"{db_size / (1024 * 1024):.2f} MB")
    else:
        st.warning(f"벡터 DB 경로가 존재하지 않습니다: {vector_db_path}")
    
    # 고급 설정
    with st.expander("고급 설정", expanded=False):
        if st.button("벡터 DB 초기화 (모든 문서 삭제)", type="secondary"):
            if st.session_state.document_manager:
                confirm = st.text_input(
                    "정말로 모든 문서를 삭제하시겠습니까? 삭제하려면 'DELETE'를 입력하세요:",
                    key="confirm_delete"
                )
                
                if confirm == "DELETE":
                    import shutil
                    try:
                        # 벡터 DB 디렉토리 삭제 및 재생성
                        shutil.rmtree(vector_db_path, ignore_errors=True)
                        os.makedirs(vector_db_path, exist_ok=True)
                        
                        # 세션 상태 초기화
                        st.session_state.document_manager = None
                        st.session_state.agent_graph = None
                        
                        st.success("벡터 DB가 초기화되었습니다. 페이지를 새로고침하세요.")
                    except Exception as e:
                        st.error(f"초기화 중 오류 발생: {e}")
        
        # 모델 설정
        st.markdown("### 모델 설정")
        
        model_option = st.selectbox(
            "사용할 OpenAI 모델:",
            options=["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo"],
            index=0,
            help="더 고급 모델은 성능이 좋지만 비용이 더 높습니다."
        )
        
        chunk_size = st.slider(
            "문서 청크 크기:",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="문서를 분할할 때 각 청크의 크기(글자 수)입니다. 너무 작으면 컨텍스트가 부족하고, 너무 크면 관련성이 떨어질 수 있습니다."
        )
        
        retrieval_count = st.slider(
            "검색 결과 수:",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="질문에 대해 벡터 DB에서 검색할 문서 수입니다."
        )
        
        if st.button("모델 설정 저장", type="primary"):
            # 설정 파일로 저장
            config = {
                "model": model_option,
                "chunk_size": chunk_size,
                "retrieval_count": retrieval_count,
                "updated_at": datetime.now().isoformat()
            }
            
            with open("agent_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            st.success("모델 설정이 저장되었습니다. 다음 실행부터 적용됩니다.")

def display_doc_browser():
    """문서 브라우저 인터페이스"""
    if not initialize_managers():
        return
    
    document_manager = st.session_state.document_manager
    
    st.markdown("## 🔍 문서 브라우저")
    
    # 문서 목록 가져오기
    docs_df = document_manager.list_documents()
    
    if docs_df.empty:
        st.info("인덱싱된 문서가 없습니다. '문서 관리' 탭에서 문서를 추가하세요.")
        return
    
    # 검색 기능
    st.markdown("### 문서 검색")
    
    search_query = st.text_input("검색어 입력:")
    search_count = st.slider("검색 결과 수:", min_value=1, max_value=10, value=3)
    
    # 태그 필터
    all_tags = set()
    for _, metadata in document_manager.document_metadata["documents"].items():
        all_tags.update(metadata.get("tags", []))
    
    selected_tags = st.multiselect(
        "태그 필터:",
        options=sorted(list(all_tags)),
        help="특정 태그가 있는 문서만 검색하려면 선택하세요."
    )
    
    filter_criteria = {"tags": {"$in": selected_tags}} if selected_tags else None
    
    if search_query and st.button("검색", type="primary"):
        with st.spinner("검색 중..."):
            results = document_manager.search_documents(
                search_query, k=search_count, filter_criteria=filter_criteria
            )
            
            if results:
                st.markdown(f"### 검색 결과: {len(results)}개 문서 찾음")
                
                for i, doc in enumerate(results):
                    with st.expander(f"문서 {i+1}: {doc.metadata.get('source', '알 수 없는 출처')}", expanded=i==0):
                        st.markdown("#### 문서 정보")
                        st.json({
                            "source": doc.metadata.get("source", "알 수 없음"),
                            "added_at": doc.metadata.get("added_at", "알 수 없음"),
                            "tags": doc.metadata.get("tags", []),
                        })
                        
                        st.markdown("#### 문서 내용")
                        st.text_area(
                            "내용",
                            value=doc.page_content,
                            height=300,
                            disabled=True
                        )
            else:
                st.warning("검색 결과가 없습니다.")
    
    # 문서 목록 브라우징
    st.markdown("### 문서 목록 브라우징")
    
    selected_document = st.selectbox(
        "문서 선택:",
        options=docs_df["filename"].tolist()
    )
    
    if selected_document:
        # 문서 해시 찾기
        selected_hash = None
        for file_hash, metadata in document_manager.document_metadata["documents"].items():
            if metadata["filename"] == selected_document:
                selected_hash = file_hash
                break
        
        if selected_hash:
            # 해당 문서의 청크 검색
            docs = document_manager.vector_store._collection.get(
                where={"file_hash": selected_hash}
            )
            
            if docs and docs["documents"]:
                st.markdown(f"### {selected_document} 청크")
                
                for i, (doc_id, content) in enumerate(zip(docs["ids"], docs["documents"])):
                    metadata = docs["metadatas"][i] if i < len(docs["metadatas"]) else {}
                    
                    with st.expander(f"청크 {i+1} (ID: {doc_id[:8]}...)", expanded=i==0):
                        st.markdown("#### 메타데이터")
                        st.json(metadata)
                        
                        st.markdown("#### 내용")
                        st.text_area(
                            "청크 내용",
                            value=content,
                            height=200,
                            disabled=True
                        )
            else:
                st.warning(f"'{selected_document}'의 청크를 찾을 수 없습니다.")

def display_chat_interface():
    """채팅 인터페이스"""
    st.markdown("## 🤖 LangGraph+RAG 에이전트")
    
    # API 키 확인
    if not st.session_state.api_keys_valid:
        st.warning("API 키가 설정되지 않았습니다. 설정 탭에서 API 키를 입력하세요.")
        return
    
    # 에이전트 초기화
    if not initialize_managers():
        return
    
    # 채팅 기록 표시
    display_chat_history()
    
    # 사용자 입력
    user_query = st.chat_input("질문을 입력하세요...")
    
    if user_query:
        send_message(user_query)
        st.rerun()  # 채팅 기록 업데이트를 위한 페이지 새로고침

def display_about():
    """소개 페이지"""
    st.markdown("## 📖 LangGraph+RAG 에이전트 소개")
    
    st.markdown("""
    ### 프로젝트 개요
    
    이 애플리케이션은 LangGraph와 RAG(Retrieval-Augmented Generation)를 결합한 고급 질의응답 시스템입니다. 
    사용자의 질문에 대해 내부 문서 지식베이스와 웹 검색을 결합하여 정확하고 근거 있는 답변을 제공합니다.
    
    ### 주요 기능
    
    - **문서 관리**: 다양한 형식(TXT, PDF, CSV, MD 등)의 문서를 지식 베이스에 추가하고 관리할 수 있습니다.
    - **RAG 질의응답**: 내부 문서에서 관련 정보를 검색하여 질문에 답변합니다.
    - **웹 검색 통합**: 필요한 경우 Google 검색 API를 통해 최신 정보를 검색합니다.
    - **대화 기록**: 질문과 답변 기록을 유지하고, 검색 및 참조 문서를 확인할 수 있습니다.
    - **문서 브라우저**: 지식 베이스 내 문서를 검색하고 탐색할 수 있습니다.
    
    ### 기술 스택
    
    - **LangGraph**: 질의응답 워크플로우를 그래프 형태로 구현하여 복잡한 프로세스를 관리합니다.
    - **LangChain**: 문서 처리, 벡터 저장소, 임베딩 등 RAG 기능을 구현합니다.
    - **OpenAI API**: 대화 생성과 문서 임베딩에 사용됩니다.
    - **Google Search API**: 웹 검색 기능을 제공합니다.
    - **Chroma DB**: 문서의 벡터 임베딩을 저장하고 검색합니다.
    - **Streamlit**: 웹 인터페이스를 구현합니다.
    
    ### 사용 방법
    
    1. **설정 탭**에서 필요한 API 키를 설정합니다.
    2. **문서 관리** 탭에서 지식 베이스에 문서를 추가합니다.
    3. **채팅** 탭에서 질문을 입력하고 답변을 받습니다.
    4. **문서 브라우저** 탭에서 지식 베이스 내 문서를 검색하고 탐색할 수 있습니다.
    
    ### 참고 자료
    
    - [LangChain 문서](https://python.langchain.com/docs/get_started/introduction)
    - [LangGraph 문서](https://python.langchain.com/docs/langgraph)
    - [RAG 아키텍처 가이드](https://www.pinecone.io/learn/retrieval-augmented-generation/)
    - [Chroma DB 문서](https://docs.trychroma.com/)
    """)
    
    # 시스템 정보
    st.markdown("### 시스템 정보")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**환경 설정**")
        
        if os.path.exists("./chroma_db"):
            db_size = sum(os.path.getsize(os.path.join(root, file)) for root, _, files in os.walk("./chroma_db") for file in files)
            st.markdown(f"- 벡터 DB 크기: {db_size / (1024 * 1024):.2f} MB")
        else:
            st.markdown("- 벡터 DB: 초기화 필요")
        
        if os.path.exists("agent_config.json"):
            with open("agent_config.json", "r") as f:
                config = json.load(f)
            st.markdown(f"- 모델: {config.get('model', 'N/A')}")
            st.markdown(f"- 청크 크기: {config.get('chunk_size', 'N/A')}")
            st.markdown(f"- 최근 설정 업데이트: {config.get('updated_at', 'N/A')}")
        else:
            st.markdown("- 모델 설정: 기본값 사용")
    
    with col2:
        st.markdown("**API 상태**")
        
        if OPENAI_API_KEY:
            st.markdown("- OpenAI API: ✅ 설정됨")
        else:
            st.markdown("- OpenAI API: ❌ 설정 필요")
        
        if GOOGLE_API_KEY and GOOGLE_CSE_ID_GENERAL:
            st.markdown("- Google Search API: ✅ 설정됨")
        else:
            st.markdown("- Google Search API: ❌ 설정 필요")

def main():
    """메인 애플리케이션"""
    st.sidebar.title("LangGraph+RAG 에이전트")
    
    # API 키 유효성 검사
    st.session_state.api_keys_valid = validate_api_keys()
    
    # 메뉴 선택
    menu = st.sidebar.radio(
        "메뉴",
        ["채팅", "문서 관리", "문서 브라우저", "설정", "소개"],
        index=0
    )
    
    # 선택된 메뉴 표시
    if menu == "채팅":
        display_chat_interface()
    elif menu == "문서 관리":
        display_document_management()
    elif menu == "문서 브라우저":
        display_doc_browser()
    elif menu == "설정":
        display_settings()
    elif menu == "소개":
        display_about()
    
    # 푸터
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 LangGraph+RAG 프로젝트")
    
    # 디버그 정보
    if st.sidebar.checkbox("디버그 정보 표시", value=False):
        st.sidebar.markdown("### 디버그 정보")
        st.sidebar.write("세션 상태:")
        debug_info = {
            "agent_graph": "초기화됨" if st.session_state.agent_graph else "초기화 안됨",
            "document_manager": "초기화됨" if st.session_state.document_manager else "초기화 안됨",
            "api_keys_valid": st.session_state.api_keys_valid,
            "chat_history_length": len(st.session_state.chat_history) if "chat_history" in st.session_state else 0
        }
        st.sidebar.json(debug_info)

if __name__ == "__main__":
    main()                                   