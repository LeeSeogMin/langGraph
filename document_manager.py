# document_manager.py
# RAG 시스템의 문서 관리 및 처리를 위한 모듈

import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader
)

class DocumentManager:
    """문서 관리 및 벡터 스토어 인덱싱을 담당하는 클래스"""
    
    def __init__(self, vector_db_path: str, openai_api_key: str):
        """
        DocumentManager 초기화
        
        Args:
            vector_db_path: 벡터 DB 저장 경로
            openai_api_key: OpenAI API 키
        """
        self.vector_db_path = vector_db_path
        self.metadata_file = os.path.join(vector_db_path, "metadata.json")
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # 벡터 DB 디렉토리 생성
        os.makedirs(vector_db_path, exist_ok=True)
        
        # 문서 메타데이터 로드 또는 초기화
        self.document_metadata = self._load_metadata()
        
        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # 벡터 스토어 초기화
        self._initialize_vector_store()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """문서 메타데이터 파일 로드"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"메타데이터 로드 오류: {e}")
                return {"documents": {}, "last_updated": datetime.now().isoformat()}
        else:
            return {"documents": {}, "last_updated": datetime.now().isoformat()}
    
    def _save_metadata(self):
        """문서 메타데이터 파일 저장"""
        self.document_metadata["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"메타데이터 저장 오류: {e}")
    
    def _initialize_vector_store(self):
        """벡터 스토어 초기화"""
        try:
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            print(f"벡터 DB 초기화 완료 (문서 수: {self.vector_store._collection.count()})")
        except Exception as e:
            print(f"벡터 DB 초기화 오류: {e}")
            # 비어있는 벡터 스토어 생성
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            self.vector_store.persist()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    def _get_document_loader(self, file_path: str):
        """파일 형식에 맞는 문서 로더 반환"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        elif ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.csv':
            return CSVLoader(file_path)
        elif ext in ['.md', '.markdown']:
            return UnstructuredMarkdownLoader(file_path)
        elif ext in ['.xls', '.xlsx']:
            return UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"지원되지 않는 파일 형식: {ext}")
    
    def add_document(self, file_path: str, tags: List[str] = None) -> bool:
        """
        새 문서를 벡터 DB에 추가
        
        Args:
            file_path: 문서 파일 경로
            tags: 문서에 할당할 태그 목록
            
        Returns:
            bool: 성공 여부
        """
        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없음: {file_path}")
            return False
        
        try:
            # 파일 해시 계산
            file_hash = self._calculate_file_hash(file_path)
            filename = os.path.basename(file_path)
            
            # 이미 존재하는 문서인지 확인
            if file_hash in self.document_metadata["documents"]:
                print(f"이미 인덱싱된 문서입니다: {filename}")
                return False
            
            # 문서 로드
            loader = self._get_document_loader(file_path)
            documents = loader.load()
            
            # 메타데이터 추가
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["source"] = filename
                doc.metadata["file_hash"] = file_hash
                doc.metadata["added_at"] = datetime.now().isoformat()
                if tags:
                    doc.metadata["tags"] = tags
            
            # 문서 분할
            split_docs = self.text_splitter.split_documents(documents)
            print(f"문서 '{filename}'를 {len(split_docs)}개 청크로 분할")
            
            # 벡터 스토어에 추가
            self.vector_store.add_documents(split_docs)
            self.vector_store.persist()
            
            # 메타데이터 업데이트
            self.document_metadata["documents"][file_hash] = {
                "filename": filename,
                "path": file_path,
                "added_at": datetime.now().isoformat(),
                "chunk_count": len(split_docs),
                "tags": tags or []
            }
            self._save_metadata()
            
            print(f"문서 '{filename}'를 성공적으로 추가했습니다.")
            return True
            
        except Exception as e:
            print(f"문서 추가 중 오류 발생: {e}")
            return False
    
    def add_directory(self, directory_path: str, tags: List[str] = None, 
                      extensions: List[str] = None) -> Dict[str, Any]:
        """
        디렉토리 내 모든 문서를 벡터 DB에 추가
        
        Args:
            directory_path: 디렉토리 경로
            tags: 문서에 할당할 태그 목록
            extensions: 처리할 파일 확장자 목록 (기본값: ['.txt', '.pdf', '.md', '.csv'])
            
        Returns:
            Dict: 성공/실패 통계
        """
        if not os.path.exists(directory_path):
            print(f"디렉토리를 찾을 수 없음: {directory_path}")
            return {"success": 0, "failed": 0, "skipped": 0}
        
        if extensions is None:
            extensions = ['.txt', '.pdf', '.md', '.csv', '.xls', '.xlsx']
        
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if not os.path.isfile(file_path):
                continue
                
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                results["skipped"] += 1
                continue
                
            success = self.add_document(file_path, tags)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
        
        print(f"디렉토리 처리 완료: {results['success']}개 성공, {results['failed']}개 실패, {results['skipped']}개 건너뜀")
        return results
    
    def remove_document(self, file_hash: str = None, filename: str = None) -> bool:
        """
        문서를 벡터 DB에서 제거
        
        Args:
            file_hash: 제거할 문서의 해시 (우선)
            filename: 제거할 문서의 파일명 (file_hash가 None인 경우)
            
        Returns:
            bool: 성공 여부
        """
        if file_hash is None and filename is None:
            print("제거할 문서의 해시 또는 파일명을 제공해야 합니다.")
            return False
        
        # 파일명으로 해시 찾기
        if file_hash is None:
            for h, metadata in self.document_metadata["documents"].items():
                if metadata["filename"] == filename:
                    file_hash = h
                    break
            
            if file_hash is None:
                print(f"'{filename}' 문서를 찾을 수 없습니다.")
                return False
        
        # 문서가 존재하는지 확인
        if file_hash not in self.document_metadata["documents"]:
            print(f"해시 '{file_hash}'에 해당하는 문서를 찾을 수 없습니다.")
            return False
        
        try:
            # 벡터 스토어에서 삭제
            self.vector_store._collection.delete(
                where={"file_hash": file_hash}
            )
            
            # 메타데이터에서 삭제
            filename = self.document_metadata["documents"][file_hash]["filename"]
            del self.document_metadata["documents"][file_hash]
            self._save_metadata()
            
            print(f"문서 '{filename}' (해시: {file_hash})를 성공적으로 제거했습니다.")
            return True
            
        except Exception as e:
            print(f"문서 제거 중 오류 발생: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 4, filter_criteria: Dict[str, Any] = None) -> List[Document]:
        """
        벡터 DB에서 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter_criteria: 필터링 기준 (예: {"tags": "finance"})
            
        Returns:
            List[Document]: 검색 결과 문서 목록
        """
        try:
            if filter_criteria:
                docs = self.vector_store.similarity_search(
                    query, k=k, filter=filter_criteria
                )
            else:
                docs = self.vector_store.similarity_search(query, k=k)
            
            print(f"쿼리 '{query}'에 대해 {len(docs)}개 문서 검색됨")
            return docs
            
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return []
    
    def list_documents(self) -> pd.DataFrame:
        """
        인덱싱된 모든 문서 목록을 데이터프레임으로 반환
        
        Returns:
            pd.DataFrame: 문서 목록
        """
        if not self.document_metadata["documents"]:
            return pd.DataFrame(columns=["filename", "added_at", "chunk_count", "tags", "hash"])
        
        docs_data = []
        for file_hash, metadata in self.document_metadata["documents"].items():
            docs_data.append({
                "filename": metadata["filename"],
                "added_at": metadata["added_at"],
                "chunk_count": metadata["chunk_count"],
                "tags": ", ".join(metadata["tags"]) if metadata["tags"] else "",
                "hash": file_hash[:8] + "..."  # 해시 일부만 표시
            })
        
        return pd.DataFrame(docs_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        벡터 DB 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        try:
            doc_count = self.vector_store._collection.count()
            
            stats = {
                "total_documents": len(self.document_metadata["documents"]),
                "total_chunks": doc_count,
                "last_updated": self.document_metadata["last_updated"],
                "file_types": {},
                "tags": {}
            }
            
            # 파일 유형 및 태그 통계
            for metadata in self.document_metadata["documents"].values():
                ext = os.path.splitext(metadata["filename"])[1].lower()
                stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                
                for tag in metadata["tags"]:
                    stats["tags"][tag] = stats["tags"].get(tag, 0) + 1
            
            return stats
            
        except Exception as e:
            print(f"통계 계산 중 오류 발생: {e}")
            return {"error": str(e)}


# 사용 예시
if __name__ == "__main__":
    # 환경 변수에서 API 키 가져오기
    from dotenv import load_dotenv
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # 문서 관리자 초기화
    doc_manager = DocumentManager(
        vector_db_path="./chroma_db",
        openai_api_key=OPENAI_API_KEY
    )
    
    # 샘플 문서 추가
    sample_dir = "./documents"
    if os.path.exists(sample_dir):
        results = doc_manager.add_directory(
            sample_dir, 
            tags=["sample", "tutorial"]
        )
        print(f"디렉토리 추가 결과: {results}")
    
    # 문서 목록 출력
    docs_df = doc_manager.list_documents()
    print("\n인덱싱된 문서 목록:")
    print(docs_df)
    
    # 검색 테스트
    if doc_manager.vector_store._collection.count() > 0:
        query = "AI 에이전트의 핵심 기능은 무엇인가요?"
        results = doc_manager.search_documents(query, k=2)
        
        print(f"\n쿼리 '{query}'에 대한 검색 결과:")
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "알 수 없는 출처")
            print(f"\n문서 {i+1} (출처: {source}):")
            print(doc.page_content[:200] + "...")