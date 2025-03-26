from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from app.services.auth_service import AuthService, oauth2_scheme
from app.models.auth import Token, User, LoginRequest
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()
auth_service = AuthService()

@router.post("/login", response_model=Token)
async def login(request: LoginRequest):
    """사용자 로그인을 처리합니다."""
    try:
        logger.info(f"로그인 시도: {request.username}")
        
        if not request.username or not request.password:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="사용자명과 비밀번호를 모두 입력해주세요."
            )
            
        user = auth_service.authenticate_user(request.username, request.password)
        token = auth_service.create_token(user)
        
        logger.info(f"로그인 성공: {request.username}")
        return token
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"로그인 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/me", response_model=User)
async def read_users_me(token: str = Depends(oauth2_scheme)):
    """현재 로그인한 사용자 정보를 반환합니다."""
    try:
        payload = auth_service.verify_token(token)
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="유효하지 않은 토큰입니다."
            )
            
        user = auth_service.users.get(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="사용자를 찾을 수 없습니다."
            )
            
        return User(**user)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"사용자 정보 조회 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 