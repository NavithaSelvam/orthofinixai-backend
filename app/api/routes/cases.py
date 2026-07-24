from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
from datetime import datetime
import uuid
from app.models.schemas import CaseCreate, CaseResponse, UserInfo
from app.api.dependencies import get_current_user
from app.db.firebase import get_db

router = APIRouter()

@router.post("/", response_model=CaseResponse)
def create_case(case: CaseCreate, current_user: UserInfo = Depends(get_current_user)):
    db = get_db()
    case_id = str(uuid.uuid4())
    
    # Verify patient belongs to doctor
    patient_doc = db.collection("patients").document(case.patient_id).get()
    if not patient_doc.exists or patient_doc.to_dict().get("doctor_id") != current_user.uid:
        raise HTTPException(status_code=403, detail="Not authorized to create case for this patient")

    case_data = case.dict()
    case_data["id"] = case_id
    case_data["status"] = "Pending Analysis"
    case_data["created_at"] = datetime.utcnow()
    
    db.collection("cases").document(case_id).set(case_data)
    return case_data

@router.get("/patient/{patient_id}", response_model=List[CaseResponse])
def get_patient_cases(patient_id: str, current_user: UserInfo = Depends(get_current_user)):
    db = get_db()
    
    # Verify patient ownership
    patient_doc = db.collection("patients").document(patient_id).get()
    if not patient_doc.exists or patient_doc.to_dict().get("doctor_id") != current_user.uid:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    docs = db.collection("cases").where("patient_id", "==", patient_id).stream()
    return [doc.to_dict() for doc in docs]

@router.post("/{case_id}/upload")
async def upload_case_image(case_id: str, file: UploadFile = File(...), current_user: UserInfo = Depends(get_current_user)):
    """
    Upload an image (OPG/Photo) for a specific case.
    Stores the image locally and links to the case.
    """
    from app.db.firebase import upload_image_to_storage
    
    db = get_db()
    
    # Read image bytes
    image_bytes = await file.read()
    
    # Upload to local storage
    image_id = str(uuid.uuid4())
    storage_url = upload_image_to_storage(image_bytes, file.filename, file.content_type or "image/jpeg")
    
    image_data = {
        "id": image_id,
        "case_id": case_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "storage_url": storage_url,
        "uploaded_at": datetime.utcnow()
    }
    
    db.collection("images").document(image_id).set(image_data)
    return {"message": "Image uploaded successfully", "image_id": image_id, "url": storage_url}
