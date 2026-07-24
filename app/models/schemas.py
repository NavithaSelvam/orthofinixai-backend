from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

class UserInfo(BaseModel):
    uid: str
    email: str
    display_name: Optional[str] = None
    role: str = "doctor"

class PatientBase(BaseModel):
    name: str
    date_of_birth: str
    gender: str
    contact_info: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    id: str
    doctor_id: str
    created_at: datetime

class CaseBase(BaseModel):
    patient_id: str
    notes: Optional[str] = None

class CaseCreate(CaseBase):
    pass

class CaseResponse(CaseBase):
    id: str
    status: str
    created_at: datetime

# New schemas for OrthofinixAI specific requirements
class PredictionResponse(BaseModel):
    prediction: str
    confidence_score: float
    recommendations: List[str]
    details: Optional[Dict[str, Any]] = None # Allowing nested clinical dictionaries

class AnalysisRecordCreate(BaseModel):
    patient_name: str
    age: int
    symptoms: List[str]
    image_url: str
    prediction: str
    confidence_score: float
    recommendations: List[str]

class AnalysisRecordResponse(AnalysisRecordCreate):
    id: str
    created_at: datetime
    
class AIReportResponse(BaseModel):
    id: str
    case_id: str
    image_url: Optional[str] = None
    abo_score: float
    arch_symmetry_score: float
    root_angulation_score: float
    andrews_score: float
    recommendations: List[str]
    details: Optional[Any] = None
    created_at: datetime

class RecalculateRequest(BaseModel):
    landmarks: Dict[str, Tuple[float, float]]
    segmented_teeth: Optional[Dict[int, Dict[str, Any]]] = None
    view_type: str = "frontal"
    bracket_pixel_width: Optional[float] = 30.0 # Default width in pixels if not scaled
    scale_factor: Optional[float] = None

