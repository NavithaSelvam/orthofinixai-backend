from fastapi import APIRouter, Depends, HTTPException
from typing import List
from datetime import datetime
import uuid
from app.models.schemas import PatientCreate, PatientResponse, UserInfo
from app.api.dependencies import get_current_user
from app.db.firebase import get_db

router = APIRouter()

@router.post("/", response_model=PatientResponse)
def create_patient(patient: PatientCreate, current_user: UserInfo = Depends(get_current_user)):
    db = get_db()
    patient_id = str(uuid.uuid4())
    
    patient_data = patient.dict()
    patient_data["id"] = patient_id
    patient_data["doctor_id"] = current_user.uid
    patient_data["created_at"] = datetime.utcnow()
    
    try:
        db.collection("patients").document(patient_id).set(patient_data)
        return patient_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[PatientResponse])
def get_patients(current_user: UserInfo = Depends(get_current_user)):
    db = get_db()
    try:
        docs = db.collection("patients").where("doctor_id", "==", current_user.uid).stream()
        patients = [doc.to_dict() for doc in docs]
        return patients
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str, current_user: UserInfo = Depends(get_current_user)):
    db = get_db()
    try:
        doc = db.collection("patients").document(patient_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Patient not found")
        data = doc.to_dict()
        if data.get("doctor_id") != current_user.uid:
            raise HTTPException(status_code=403, detail="Not authorized to view this patient")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
