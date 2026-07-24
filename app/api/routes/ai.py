from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
import uuid
from datetime import datetime
from typing import Optional
from app.models.schemas import AIReportResponse, UserInfo, RecalculateRequest
from app.api.dependencies import get_current_user
from app.db.firebase import get_db, upload_image_to_storage
from app.services.ai_engine import ai_engine

router = APIRouter()

@router.post("/analyze/{case_id}", response_model=AIReportResponse)
async def analyze_case_image(
    case_id: str,
    file: UploadFile = File(...),
    view_type: str = Form("frontal"),
    bracket_pixel_width: float = Form(30.0),
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Upload an image and run it through the real AI prediction system immediately.
    Supports view_type and bracket width calibration.
    """
    db = get_db()
    
    # 1. Read image bytes
    contents = await file.read()
    
    # 2. Upload image to Storage (local file mock)
    file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    unique_filename = f"{case_id}_{uuid.uuid4()}.{file_extension}"
    image_url = upload_image_to_storage(contents, unique_filename, file.content_type)
    
    try:
        # 3. Run AI Engine
        analysis_results = ai_engine.analyze_image(
            contents, 
            view_type=view_type, 
            bracket_pixel_width=bracket_pixel_width
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error analyzing image: {str(e)}")

    # 4. Store the result
    report_id = str(uuid.uuid4())
    report_data = {
        "id": report_id,
        "case_id": case_id,
        "image_url": image_url,
        "abo_score": analysis_results["abo_score"],
        "arch_symmetry_score": analysis_results["arch_symmetry_score"],
        "root_angulation_score": analysis_results["root_angulation_score"],
        "andrews_score": analysis_results["andrews_score"],
        "recommendations": analysis_results["recommendations"],
        "details": analysis_results.get("details", {}),
        "created_at": datetime.utcnow()
    }
    
    db.collection("ai_reports").document(report_id).set(report_data)
    
    # Update case status
    db.collection("cases").document(case_id).set(
        {"status": "Analysis Complete", "updated_at": datetime.utcnow()}, 
        merge=True
    )
    
    return report_data

@router.get("/report/{case_id}", response_model=AIReportResponse)
def get_case_report(case_id: str, current_user: UserInfo = Depends(get_current_user)):
    db = get_db()
    
    # Get the latest report for this case
    docs = db.collection("ai_reports").where("case_id", "==", case_id).order_by("created_at", direction="DESCENDING").limit(1).stream()
    
    reports = [doc.to_dict() for doc in docs]
    if not reports:
        raise HTTPException(status_code=404, detail="No AI report found for this case")
        
    return reports[0]

@router.post("/recalculate")
def recalculate_metrics(request: RecalculateRequest, current_user: UserInfo = Depends(get_current_user)):
    """
    Recalculates Andrews Six Keys and other diagnostic metrics using manually adjusted coordinates.
    """
    try:
        results = ai_engine.recalculate_from_landmarks(
            landmarks=request.landmarks,
            segmented_teeth=request.segmented_teeth,
            view_type=request.view_type,
            bracket_pixel_width=request.bracket_pixel_width,
            scale_factor=request.scale_factor
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

