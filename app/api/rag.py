from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Dict, Any
from app.services.rag_service import RAGService
from fastapi.responses import JSONResponse
from app.services.auth_service import oauth2_scheme
from app.models.chat import ChatRequest, ChatResponse
from app.core.config import settings
from app.utils.logger import setup_logger
from urllib.parse import unquote
from contextlib import asynccontextmanager

logger = setup_logger(__name__)

class RAGRouter:
    def __init__(self):
        self.router = APIRouter(lifespan=self.lifespan)
        self.rag_service = None
        self._setup_routes()

    @asynccontextmanager
    async def lifespan(self, router: APIRouter):
        try:
            self.rag_service = RAGService(model=settings.RAG_MODEL)
            logger.info("RAG 서비스 초기화 완료")
        except Exception as e:
            logger.error(f"RAG 서비스 초기화 중 오류 발생: {str(e)}")
            raise
        yield

    def _setup_routes(self):
        @self.router.post("/files")
        async def get_files():
            """업로드된 PDF 파일 목록을 반환합니다."""
            try:
                logger.info("파일 목록 요청 처리 시작")
                
                if not self.rag_service:
                    raise HTTPException(status_code=503, detail="RAG 서비스가 초기화되지 않았습니다.")
                    
                files = self.rag_service.get_file_list()
                logger.info(f"파일 목록 조회 완료: {len(files)}개 파일 발견")
                
                return JSONResponse(
                    content={
                        "status": "success",
                        "files": files
                    },
                    status_code=200
                )
            except Exception as e:
                logger.error(f"파일 목록 조회 중 오류 발생: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.delete("/delete/{filename}")
        async def delete_file(filename: str, token: str = Depends(oauth2_scheme)):
            """PDF 파일을 삭제합니다."""
            try:
                decoded_filename = unquote(filename)
                logger.info(f"파일 삭제 요청: {decoded_filename}")
                
                if not self.rag_service:
                    raise HTTPException(status_code=503, detail="RAG 서비스가 초기화되지 않았습니다.")
                    
                if decoded_filename not in self.rag_service.metadata.get("files", {}):
                    raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {decoded_filename}")
                    
                success = self.rag_service.delete_file(decoded_filename)
                if success:
                    logger.info(f"파일 삭제 완료: {decoded_filename}")
                    return JSONResponse(
                        content={
                            "status": "success",
                            "message": f"{decoded_filename} 삭제 완료"
                        }
                    )
                raise HTTPException(status_code=500, detail=f"파일 삭제 중 오류가 발생했습니다: {decoded_filename}")
                
            except HTTPException as he:
                raise he
            except Exception as e:
                logger.error(f"파일 삭제 중 오류 발생: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/upload")
        async def upload_pdfs(
            file: UploadFile = File(...),
            token: str = Depends(oauth2_scheme)
        ):
            """PDF 파일을 업로드하고 인덱싱합니다."""
            try:
                logger.info(f"파일 업로드 요청 시작: {file.filename}")
                logger.info(f"Content-Type: {file.content_type}")
                
                if not file:
                    logger.error("파일이 업로드되지 않았습니다.")
                    raise HTTPException(status_code=400, detail="파일이 업로드되지 않았습니다.")
                
                if not file.filename.endswith('.pdf'):
                    logger.error(f"잘못된 파일 형식: {file.filename}")
                    raise HTTPException(status_code=400, detail=f"{file.filename}는 PDF 파일이 아닙니다.")
                
                content = await file.read()
                if not content:
                    logger.error(f"비어있는 파일: {file.filename}")
                    raise HTTPException(status_code=400, detail=f"{file.filename}는 비어있는 파일입니다.")
                
                if not self.rag_service:
                    logger.error("RAG 서비스가 초기화되지 않았습니다.")
                    raise HTTPException(status_code=503, detail="RAG 서비스가 초기화되지 않았습니다.")
                    
                logger.info("문서 인덱싱 시작")
                self.rag_service.index_documents([content], [file.filename])
                logger.info("문서 인덱싱 완료")
                
                return JSONResponse(
                    content={
                        "status": "success",
                        "message": "파일 업로드 및 인덱싱 완료",
                        "files": [file.filename]
                    }
                )
            except HTTPException as he:
                logger.error(f"HTTP 예외 발생: {str(he.detail)}")
                raise he
            except Exception as e:
                logger.error(f"파일 업로드 중 예상치 못한 오류 발생: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/ask", response_model=ChatResponse)
        async def ask(request: ChatRequest):
            """채팅 질문에 대한 답변을 생성합니다."""
            try:
                logger.info(f"채팅 요청: {request.query}")
                
                if not self.rag_service:
                    raise HTTPException(status_code=503, detail="RAG 서비스가 초기화되지 않았습니다.")
                
                response = self.rag_service.query(request.query)
                return ChatResponse(
                    answer=response,
                    model=request.model
                )
            except HTTPException as he:
                raise he
            except Exception as e:
                logger.error(f"채팅 처리 중 오류 발생: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

router = RAGRouter().router
