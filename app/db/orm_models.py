from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.sqlalchemy_db import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users_orm"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=True)
    display_name = Column(String, default="Doctor")
    role = Column(String, default="doctor")
    created_at = Column(DateTime, default=utcnow)

    analyses = relationship("AnalysisReport", back_populates="user")


class AnalysisReport(Base):
    __tablename__ = "analysis_reports"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users_orm.id"), nullable=False, index=True)
    patient_name = Column(String, default="Patient")
    image_url = Column(String, nullable=True)
    view_type = Column(String, default="frontal")
    status = Column(String, default="completed")

    finishing_score = Column(Float, default=0.0)
    alignment_score = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    midline_deviation_mm = Column(Float, default=0.0)
    overjet_mm = Column(Float, default=0.0)
    overbite_percent = Column(Float, default=0.0)
    abo_score = Column(Float, default=0.0)
    andrews_score = Column(Float, default=0.0)

    prediction = Column(Text, default="")
    recommendations_json = Column(Text, default="[]")
    metrics_json = Column(Text, default="{}")

    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="analyses")
