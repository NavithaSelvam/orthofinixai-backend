import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "orthofinix_summit.db")
)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_sqlalchemy():
    from app.db import orm_models  # noqa: F401
    Base.metadata.create_all(bind=engine)
