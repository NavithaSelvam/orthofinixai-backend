from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    display_name: str = "Doctor"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class UploadResponse(BaseModel):
    upload_id: str
    image_url: str
    filename: str


class AnalyzeRequest(BaseModel):
    upload_id: Optional[str] = None
    patient_name: str = "Patient"
    view_type: str = "frontal"


class MetricCard(BaseModel):
    label: str
    value: str
    severity: str = "normal"
    ideal: Optional[str] = None


class AnalysisReportResponse(BaseModel):
    id: str
    patient_name: str
    image_url: Optional[str] = None
    view_type: str
    status: str
    finishing_score: float
    alignment_score: float
    confidence_score: float
    midline_deviation_mm: float
    overjet_mm: float
    overbite_percent: float
    abo_score: float
    andrews_score: float
    prediction: str
    recommendations: List[str]
    metrics: Dict[str, Any]
    created_at: datetime


class HistoryItem(BaseModel):
    id: str
    patient_name: str
    finishing_score: float
    confidence_score: float
    created_at: datetime
    image_url: Optional[str] = None
