import uuid
from sqlalchemy.orm import Session
from app.db.orm_models import User
from app.core.security import hash_password, verify_password


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email.lower()).first()


def create_user(db: Session, email: str, password: str, display_name: str) -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=email.lower(),
        password_hash=hash_password(password),
        display_name=display_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if user and verify_password(password, user.password_hash):
        return user
    return None
