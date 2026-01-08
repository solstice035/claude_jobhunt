from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from jose import JWTError, jwt
from app.config import get_settings

settings = get_settings()

ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30
COOKIE_NAME = "session_token"


def create_session_token() -> str:
    expire = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    to_encode = {"exp": expire, "authenticated": True}
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)


def verify_session_token(token: str) -> bool:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload.get("authenticated", False)
    except JWTError:
        return False


def verify_password(password: str) -> bool:
    return password == settings.app_password


async def get_current_user(request: Request) -> bool:
    token = request.cookies.get(COOKIE_NAME)
    if not token or not verify_session_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return True
