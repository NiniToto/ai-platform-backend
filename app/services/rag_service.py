import os
import json
import pdfplumber
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from llama_index.core.schema import Document
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import faiss
from PIL import Image
import io
import base64
import shutil
import gc
from threading import Lock
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
from app.services.prompt_service import PromptService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGService:
    def __init__(self, model: str = "Llama-3.1"):
        self.model_name = model
        self.prompt_service = PromptService()
        self.executor = ThreadPoolExecutor(max_workers=4)  # 스레드 풀 생성
        
        # 락 초기화
        self.metadata_lock = Lock()
        self.index_lock = Lock()
        
        # 모델 설정
        if model == "Gemma-3":
            self.llm = Ollama(model="gemma3:12b", request_timeout=120.0)
        elif model == "Llama-3.1":
            self.llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
        else:
            self.llm = Ollama(model="deepseek-r1:8b", request_timeout=120.0)

        Settings.llm = self.llm
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            trust_remote_code=True,
            device="cuda"  # GPU 사용
        )
        
        # 파일 저장 디렉토리 설정
        self.base_dir = "uploaded_files"
        self.index_dir = "vector_store"
        self.metadata_file = os.path.join(self.base_dir, "metadata.json")
        self.chunks_dir = "chunks_data"
        self.archive_dir = "chunks_archive"
        
        # 디렉토리 생성 (한 번에 처리)
        for dir_path in [self.base_dir, self.index_dir, self.chunks_dir, self.archive_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 메타데이터 로드
        self.metadata = self._load_metadata()
        
        # 지연 로딩을 위한 초기화
        self.embed_model = None
        self.node_parser = None
        self.index = None
        self.query_engine = None
        
        # 인덱스와 쿼리 엔진 초기화
        self._load_or_create_index()
        
        # 청크 데이터 정리 (비동기로 실행)
        asyncio.create_task(self._cleanup_old_chunks())

    def __del__(self):
        # 리소스 정리
        self.executor.shutdown(wait=True)
        gc.collect()

    @lru_cache(maxsize=100)
    def _get_embedding(self, text: str):
        """임베딩 결과를 캐싱합니다."""
        if self.embed_model is None:
            self._initialize_models()
        return self.embed_model.get_text_embedding(text)

    def _initialize_models(self):
        """필요할 때만 모델을 초기화합니다."""
        if self.embed_model is None:
            # CPU 임베딩 모델 사용
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-large-en-v1.5",
                trust_remote_code=True,
                device="cuda"  # GPU 사용
            )
            Settings.embed_model = self.embed_model
            
            self.node_parser = SentenceSplitter(
                chunk_size=256,  # 청크 크기 축소
                chunk_overlap=25,  # 오버랩 축소
                paragraph_separator="\n\n",
                secondary_chunking_regex=r"(?<=\. )",
                include_metadata=True,
                include_prev_next_rel=True
            )
            Settings.node_parser = self.node_parser

    def _extract_text_from_pdf(self, file_path: str) -> List[str]:
        """PDF 파일에서 텍스트를 추출하고 청크로 분할합니다."""
        text_chunks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                print(f"PDF 페이지 수: {len(pdf.pages)}")
                # 배치 처리로 변경
                batch_size = 5
                for i in range(0, len(pdf.pages), batch_size):
                    batch = pdf.pages[i:i + batch_size]
                    for page_num, page in enumerate(batch, start=i):
                        # 1. 일반 텍스트 추출 시도
                        text = page.extract_text()
                        
                        # 2. 텍스트가 없는 경우 다른 방법 시도
                        if not text or not text.strip():
                            print(f"페이지 {page_num + 1}에서 일반 텍스트 추출 실패, 대체 방법 시도")
                            tables = page.extract_tables()
                            if tables:
                                table_texts = []
                                for table in tables:
                                    if table:
                                        table_text = "\n".join([
                                            " | ".join([str(cell) if cell else "" for cell in row])
                                            for row in table
                                        ])
                                        table_texts.append(table_text)
                                text = "\n\n".join(table_texts)
                        
                        # 3. 여전히 텍스트가 없는 경우
                        if not text or not text.strip():
                            print(f"페이지 {page_num + 1}에서 모든 텍스트 추출 방법 실패")
                            continue
                        
                        # 텍스트 전처리
                        text = text.strip()
                        text = " ".join(text.split())
                        
                        if text:
                            text_with_metadata = f"[페이지 {page_num + 1}] {text}"
                            text_chunks.append(text_with_metadata)
                            print(f"페이지 {page_num + 1}에서 텍스트 추출 완료: {len(text)}자")
                        else:
                            print(f"페이지 {page_num + 1}에서 텍스트를 추출할 수 없습니다.")
                    
                    # 배치 처리 후 메모리 정리
                    gc.collect()
            
            print(f"총 {len(text_chunks)}개의 청크 생성됨")
            return text_chunks
            
        except Exception as e:
            print(f"PDF 텍스트 추출 중 오류 발생: {str(e)}")
            return []
        finally:
            # 메모리 정리
            gc.collect()

    def _save_chunks(self, file_name: str, chunks: List[str]) -> str:
        """청크 데이터를 저장하고 경로를 반환합니다."""
        try:
            base_name = os.path.splitext(file_name)[0]
            chunks_file = os.path.join(self.chunks_dir, f"{base_name}_chunks.json")
            
            chunks_data = {
                "chunks": chunks,
                "created_at": str(datetime.now()),
                "total_chunks": len(chunks)
            }
            
            # JSON 데이터를 파일로 저장
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            
            return chunks_file
        except Exception as e:
            print(f"청크 데이터 저장 중 오류 발생: {str(e)}")
            raise

    def _load_chunks(self, chunks_file: str) -> List[str]:
        """청크 데이터를 로드합니다."""
        try:
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    return chunks_data.get("chunks", [])
            return []
        except Exception as e:
            print(f"청크 데이터 로드 중 오류 발생: {str(e)}")
            return []

    def _load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # 기존 메타데이터 구조를 새로운 구조로 변환
                    if "files" not in metadata:
                        metadata["files"] = {}
                    if "last_updated" not in metadata:
                        metadata["last_updated"] = str(datetime.now())
                    # chunks 필드가 있다면 제거
                    if "chunks" in metadata:
                        del metadata["chunks"]
                    return metadata
            except Exception as e:
                print(f"메타데이터 로드 중 오류 발생: {str(e)}")
                # 오류 발생 시 새로운 구조로 초기화
                return {
                    "files": {},
                    "last_updated": str(datetime.now())
                }
        return {
            "files": {},
            "last_updated": str(datetime.now())
        }

    def _save_metadata(self):
        """메타데이터 저장 시 락을 사용합니다."""
        with self.metadata_lock:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _load_or_create_index(self):
        """저장된 인덱스를 로드하거나 새로운 인덱스를 생성합니다."""
        with self.index_lock:
            if self.index is not None:
                return
                
            self._initialize_models()
            
            index_path = os.path.join(self.index_dir, "index.faiss")
            documents = []
            
            # 메타데이터에서 문서 정보 로드
            for file_name, info in self.metadata.get("files", {}).items():
                chunks_file = info.get("chunks_file")
                if chunks_file and os.path.exists(chunks_file):
                    text_chunks = self._load_chunks(chunks_file)
                    for i, chunk in enumerate(text_chunks):
                        chunk_id = f"{file_name}_chunk_{i}"
                        doc = Document(
                            text=chunk,
                            metadata={
                                "file_name": file_name,
                                "file_path": info["path"],
                                "chunk_id": chunk_id
                            }
                        )
                        documents.append(doc)
            
            if os.path.exists(index_path):
                # 저장된 인덱스 로드
                dimension = 1024
                faiss_index = faiss.read_index(index_path)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self.index = VectorStoreIndex(documents, storage_context=storage_context)
            else:
                # 새로운 인덱스 생성
                dimension = 1024
                faiss_index = faiss.IndexFlatL2(dimension)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self.index = VectorStoreIndex(documents, storage_context=storage_context)

            # 검색기 설정 - 검색 결과 수와 모드 최적화
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3,  # 검색 결과 수 감소
                vector_store_query_mode="default",  # 기본 모드 사용
                filters=None
            )
            
            # 유사도 임계값 조정
            node_postprocessors = [
                SimilarityPostprocessor(similarity_cutoff=0.3)  # 임계값 상향 조정
            ]
            
            # 쿼리 엔진 생성 - 응답 모드 최적화
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=node_postprocessors,
                response_mode="tree_summarize",  # 응답 모드 변경
                response_kwargs={
                    "verbose": False,
                    "response_template": "{response}"  # 응답 템플릿 설정
                }
            )
            
            # 메모리 정리
            gc.collect()

    async def _cleanup_old_chunks(self):
        """30일 이상 된 청크 데이터를 보관소로 이동합니다."""
        try:
            current_time = datetime.now()
            for filename in os.listdir(self.chunks_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.chunks_dir, filename)
                    file_stat = os.stat(file_path)
                    file_time = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    if current_time - file_time > timedelta(days=30):
                        archive_path = os.path.join(self.archive_dir, filename)
                        # 파일 이동 후 원본 삭제
                        shutil.move(file_path, archive_path)
                        print(f"오래된 청크 데이터 보관: {filename}")
            
            # 보관소 정리 (90일 이상 된 데이터 삭제)
            for filename in os.listdir(self.archive_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.archive_dir, filename)
                    file_stat = os.stat(file_path)
                    file_time = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    if current_time - file_time > timedelta(days=90):
                        os.remove(file_path)
                        print(f"오래된 보관 데이터 삭제: {filename}")
            
            # 메모리 정리
            gc.collect()
            
        except Exception as e:
            print(f"청크 데이터 정리 중 오류 발생: {str(e)}")

    def index_documents(self, uploaded_files: List[bytes], file_names: List[str]):
        try:
            documents = []
            
            for file_data, name in zip(uploaded_files, file_names):
                print(f"파일 업로드 처리 중: {name}")
                
                # 파일 저장
                file_path = os.path.join(self.base_dir, name)
                try:
                    with open(file_path, "wb") as f:
                        f.write(file_data)
                    print(f"파일 저장 완료: {file_path}")
                except Exception as e:
                    print(f"파일 저장 중 오류 발생: {str(e)}")
                    continue
                
                # 파일이 실제로 저장되었는지 확인
                if not os.path.exists(file_path):
                    print(f"파일이 저장되지 않았습니다: {file_path}")
                    continue
                
                # PDF에서 텍스트 추출 및 청크 분할
                text_chunks = self._extract_text_from_pdf(file_path)
                print(f"추출된 청크 수: {len(text_chunks)}")
                
                if not text_chunks:
                    print(f"경고: {name}에서 텍스트를 추출할 수 없습니다.")
                    continue
                
                # 청크 데이터 저장
                try:
                    chunks_file = self._save_chunks(name, text_chunks)
                    print(f"청크 데이터 저장 완료: {chunks_file}")
                except Exception as e:
                    print(f"청크 데이터 저장 중 오류 발생: {str(e)}")
                    continue
                
                # 문서 생성 및 메타데이터 추가
                for i, chunk in enumerate(text_chunks):
                    chunk_id = f"{name}_chunk_{i}"
                    doc = Document(
                        text=chunk,
                        metadata={
                            "file_name": name,
                            "file_path": file_path,
                            "chunk_id": chunk_id
                        }
                    )
                    documents.append(doc)
                
                # 메타데이터 업데이트
                try:
                    self.metadata["files"][name] = {
                        "path": file_path,
                        "uploaded_at": str(datetime.now()),
                        "size": len(file_data),
                        "chunk_count": len(text_chunks),
                        "chunks_file": chunks_file
                    }
                    print(f"메타데이터 업데이트 완료: {name}")
                except Exception as e:
                    print(f"메타데이터 업데이트 중 오류 발생: {str(e)}")
                    continue
            
            # 메타데이터 저장
            try:
                self.metadata["last_updated"] = str(datetime.now())
                self._save_metadata()
                print("메타데이터 저장 완료")
            except Exception as e:
                print(f"메타데이터 저장 중 오류 발생: {str(e)}")

            # 인덱스 생성
            if documents:
                print(f"총 {len(documents)}개의 문서로 인덱스 생성")
                dimension = 1024
                faiss_index = faiss.IndexFlatL2(dimension)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                self.index = VectorStoreIndex(documents, storage_context=storage_context)
                print("인덱스 생성 완료")
                
                # 인덱스 저장
                index_path = os.path.join(self.index_dir, "index.faiss")
                try:
                    faiss.write_index(faiss_index, index_path)
                    print("FAISS 인덱스 저장 완료")
                except Exception as e:
                    print(f"FAISS 인덱스 저장 중 오류 발생: {str(e)}")
                
                # 검색기 설정
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=3,  # 검색 결과 수 감소
                    vector_store_query_mode="default",  # 기본 모드 사용
                    filters=None
                )
                
                node_postprocessors = [
                    SimilarityPostprocessor(similarity_cutoff=0.3)  # 임계값 상향 조정
                ]
                
                self.query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    node_postprocessors=node_postprocessors,
                    response_mode="tree_summarize",  # 응답 모드 변경
                    response_kwargs={
                        "verbose": False,
                        "response_template": "{response}"  # 응답 템플릿 설정
                    }
                )
                print("쿼리 엔진 설정 완료")
            else:
                print("처리할 문서가 없습니다.")
                
        except Exception as e:
            print(f"문서 인덱싱 중 오류 발생: {str(e)}")
            raise

    def query(self, query: str) -> str:
        """문서에 대한 질문에 답변합니다."""
        try:
            if not self.query_engine:
                raise ValueError("쿼리 엔진이 초기화되지 않았습니다.")
            
            # 프롬프트 생성
            structured_prompt = self.prompt_service.create_structured_prompt(query)
            
            # 문서 검색 및 답변 생성
            logger.info(f"쿼리 실행: {query}")
            response = self.query_engine.query(structured_prompt)
            
            # 답변 후처리
            processed_response = self._post_process_response(str(response))
            logger.info(f"답변 후처리 완료: {processed_response}")
            
            return processed_response
            
        except Exception as e:
            logger.error(f"쿼리 처리 중 오류 발생: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _post_process_response(self, response: str) -> str:
        """답변을 후처리하여 더 구조화된 형태로 만듭니다."""
        try:
            # 기본 구조 추가
            structured_response = f"""답변:

{response}

---
참고: 이 답변은 업로드된 문서들을 기반으로 생성되었습니다."""
            
            return structured_response
            
        except Exception as e:
            logger.error(f"답변 후처리 중 오류 발생: {str(e)}")
            return response  # 후처리 실패시 원본 응답 반환

    def get_file_list(self) -> List[Dict]:
        """업로드된 파일 목록을 반환합니다."""
        try:
            # 메타데이터가 없거나 files 키가 없는 경우 빈 리스트 반환
            if not self.metadata or "files" not in self.metadata:
                print("메타데이터가 없거나 files 키가 없습니다.")
                return []
                
            return [
                {
                    "name": name,
                    "uploaded_at": info.get("uploaded_at", ""),
                    "size": info.get("size", 0),
                    "llm_model": self.model_name,  # LLM 모델 이름
                    "embedding_model": "BGE-large-en-v1.5"  # 임베딩 모델 이름
                }
                for name, info in self.metadata["files"].items()
            ]
        except Exception as e:
            print(f"파일 목록 조회 중 오류 발생: {str(e)}")
            return []

    def delete_file(self, file_name: str) -> bool:
        """파일을 삭제하고 관련 메타데이터와 인덱스를 업데이트합니다."""
        try:
            print(f"파일 삭제 시작: {file_name}")  # 디버깅 로그
            
            # 메타데이터 검증
            if not self.metadata or "files" not in self.metadata:
                print("메타데이터가 없거나 files 키가 없습니다.")
                return False
                
            if file_name not in self.metadata["files"]:
                print(f"파일 '{file_name}'이(가) 메타데이터에 없습니다.")
                return False
                
            # 파일 경로 가져오기
            file_path = self.metadata["files"][file_name]["path"]
            chunks_file = self.metadata["files"][file_name].get("chunks_file")
            
            print(f"삭제할 파일 경로: {file_path}")  # 디버깅 로그
            print(f"삭제할 청크 파일: {chunks_file}")  # 디버깅 로그
            
            # 파일 삭제
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"파일 '{file_path}' 삭제 완료")
                except Exception as e:
                    print(f"파일 삭제 중 오류 발생: {str(e)}")
                    return False
            
            # 청크 데이터 파일 삭제
            if chunks_file and os.path.exists(chunks_file):
                try:
                    os.remove(chunks_file)
                    print(f"청크 데이터 파일 '{chunks_file}' 삭제 완료")
                except Exception as e:
                    print(f"청크 데이터 파일 삭제 중 오류 발생: {str(e)}")
            
            # 메타데이터에서 파일 정보 삭제
            del self.metadata["files"][file_name]
            
            # 메타데이터 저장
            self.metadata["last_updated"] = str(datetime.now())
            try:
                self._save_metadata()
                print("메타데이터 저장 완료")
            except Exception as e:
                print(f"메타데이터 저장 중 오류 발생: {str(e)}")
                return False
            
            # 인덱스 재생성
            try:
                self._recreate_index()
                print("인덱스 재생성 완료")
            except Exception as e:
                print(f"인덱스 재생성 중 오류 발생: {str(e)}")
                return False
            
            print(f"파일 '{file_name}' 삭제 완료")  # 디버깅 로그
            return True
            
        except Exception as e:
            print(f"파일 삭제 중 예상치 못한 오류 발생: {str(e)}")
            return False

    def _recreate_index(self):
        """인덱스를 재생성합니다."""
        try:
            print("인덱스 재생성 시작")
            # 새로운 인덱스 생성
            dimension = 1024
            faiss_index = faiss.IndexFlatL2(dimension)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 남은 파일들로 인덱스 재구성
            documents = []
            for file_name, info in self.metadata["files"].items():
                file_path = info["path"]
                chunks_file = info.get("chunks_file")
                
                if os.path.exists(file_path) and chunks_file:
                    print(f"파일 처리 중: {file_name}")
                    # 저장된 청크 데이터 로드
                    text_chunks = self._load_chunks(chunks_file)
                    print(f"로드된 청크 수: {len(text_chunks)}")
                    
                    for i, chunk in enumerate(text_chunks):
                        chunk_id = f"{file_name}_chunk_{i}"
                        doc = Document(
                            text=chunk,
                            metadata={
                                "file_name": file_name,
                                "file_path": file_path,
                                "chunk_id": chunk_id
                            }
                        )
                        documents.append(doc)
            
            if documents:
                print(f"총 {len(documents)}개의 문서로 인덱스 생성")
                self.index = VectorStoreIndex(documents, storage_context=storage_context)
                
                # 인덱스 저장
                index_path = os.path.join(self.index_dir, "index.faiss")
                try:
                    faiss.write_index(faiss_index, index_path)
                    print("FAISS 인덱스 저장 완료")
                except Exception as e:
                    print(f"FAISS 인덱스 저장 중 오류 발생: {str(e)}")
                
                # 검색기 재설정
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=3,  # 검색 결과 수 감소
                    vector_store_query_mode="default",  # 기본 모드 사용
                    filters=None
                )
                
                node_postprocessors = [
                    SimilarityPostprocessor(similarity_cutoff=0.3)  # 임계값 상향 조정
                ]
                
                self.query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    node_postprocessors=node_postprocessors,
                    response_mode="tree_summarize",  # 응답 모드 변경
                    response_kwargs={
                        "verbose": False,
                        "response_template": "{response}"  # 응답 템플릿 설정
                    }
                )
                print("쿼리 엔진 재설정 완료")
            else:
                print("처리할 문서가 없습니다.")
                
        except Exception as e:
            print(f"인덱스 재생성 중 오류 발생: {str(e)}")
            raise
