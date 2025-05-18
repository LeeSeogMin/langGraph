# LangGraph + RAG 에이전트

LangGraph와 RAG(Retrieval-Augmented Generation)를 결합한 고급 AI 에이전트 시스템입니다. 이 애플리케이션은 문서 검색, 웹 검색, 그리고 LLM을 활용한 질의응답 기능을 제공합니다.

## 주요 기능

- **문서 관리**: 다양한 형식(PDF, TXT, CSV, Markdown 등)의 문서 추가 및 관리
- **벡터 검색**: 의미 기반 문서 검색을 위한 벡터 DB 활용
- **웹 검색 통합**: Google 검색 및 Google Scholar 연동
- **LangGraph 기반 에이전트**: 복잡한 추론 및 의사결정 프로세스를 위한 그래프 기반 아키텍처

## 설치 방법

1. 가상 환경 생성 및 활성화:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

2. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

3. `.env` 파일 설정:
   `.env` 파일에 다음과 같이 API 키를 입력해주세요:
   ```
   # OpenAI API 키
   OPENAI_API_KEY=your_openai_api_key_here

   # Google 검색 API 설정
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_CSE_ID_GENERAL=your_google_cse_id_here
   GOOGLE_CSE_ID_SCHOLAR=your_google_scholar_cse_id_here
   ```

## 실행 방법

Streamlit 웹 인터페이스 실행:
```bash
streamlit run app.py
```

## 파일 구조

- `app.py`: Streamlit 웹 인터페이스
- `document_manager.py`: 문서 관리 및 벡터 DB 처리 모듈
- `rag_agent.py`: LangGraph와 RAG를 결합한 에이전트 구현
- `agent.py`: 기본 에이전트 구현 (보조)
- `chroma_db/`: 벡터 데이터베이스 저장 디렉토리

## 참고 사항

- 이 애플리케이션을 사용하려면 OpenAI API 키가 필요합니다.
- Google 검색 기능을 사용하려면 Google API 키와 Custom Search Engine ID가 필요합니다.
- 기본 모델은 gpt-3.5-turbo-16k이며, agent_config.json 파일에서 변경할 수 있습니다. 