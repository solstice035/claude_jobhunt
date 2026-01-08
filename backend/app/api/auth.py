from fastapi import APIRouter, Response, HTTPException, status
from app.schemas import LoginRequest, LoginResponse
from app.auth import verify_password, create_session_token, COOKIE_NAME

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, response: Response):
    if not verify_password(request.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )

    token = create_session_token()
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        max_age=30 * 24 * 60 * 60,  # 30 days
        samesite="lax",
    )
    return LoginResponse(success=True, message="Logged in successfully")


@router.post("/logout", response_model=LoginResponse)
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return LoginResponse(success=True, message="Logged out successfully")


@router.get("/check")
async def check_auth():
    return {"authenticated": True}
