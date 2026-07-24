from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.db.sqlalchemy_db import get_db_session
from app.models.summit_schemas import RegisterRequest, LoginRequest, TokenResponse
from app.services.user_service import create_user, authenticate_user, get_user_by_email

router = APIRouter()


@router.post("/register", response_model=TokenResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db_session)):
    if get_user_by_email(db, body.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    user = create_user(db, body.email, body.password, body.display_name)
    token = create_access_token(user.id, user.email, user.display_name)
    return TokenResponse(
        access_token=token,
        user={"id": user.id, "email": user.email, "display_name": user.display_name},
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db_session)):
    user = authenticate_user(db, body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user.id, user.email, user.display_name)
    return TokenResponse(
        access_token=token,
        user={"id": user.id, "email": user.email, "display_name": user.display_name},
    )
