from datetime import timedelta
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from app.core.security import create_access_token, verify_token
from app.core.config import settings
from app.models.auth import Token, User
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthService:
    def __init__(self):
        self.users = {
            "admin": {
                "username": "admin",
                "password": pwd_context.hash("*****"),  # 비밀번호 해시화
                "disabled": False,
                "role": "admin"
            }
        }

    def authenticate_user(self, username: str, password: str) -> User:
        """사용자를 인증합니다."""
        user = self.users.get(username)
        if not user or not pwd_context.verify(password, user["password"]):
            logger.warning(f"인증 실패: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="잘못된 사용자명 또는 비밀번호입니다.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        logger.info(f"인증 성공: {username}")
        return User(**user)

    def create_token(self, user: User) -> Token:
        """액세스 토큰을 생성합니다."""
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "role": user.role},
            expires_delta=access_token_expires
        )
        logger.info(f"토큰 생성 완료: {user.username}")
        return Token(access_token=access_token, token_type="bearer")

    def verify_token(self, token: str) -> dict:
        """토큰을 검증합니다."""
        try:
            payload = verify_token(token)
            logger.info(f"토큰 검증 성공: {payload.get('sub')}")
            return payload
        except Exception as e:
            logger.error(f"토큰 검증 실패: {str(e)}")
            raise 
