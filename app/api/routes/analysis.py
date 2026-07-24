from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import List
from app.models.schemas import (
    PredictionResponse,
    AnalysisRecordCreate,
    AnalysisRecordResponse,
    UserInfo,
)
from app.services.ai_engine import ai_engine
from app.db.firebase import (
    save_analysis_record,
    get_user_analysis_history,
    get_analysis_by_id,
    upload_image_to_storage,
)
from app.api.dependencies import get_current_user
import uuid
import os
import traceback

router = APIRouter(prefix="/analysis")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Uploads an image and returns upload_id and image_url.
    """

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()

        file_extension = (
            file.filename.split(".")[-1]
            if "." in file.filename
            else "jpg"
        )

        unique_filename = f"{uuid.uuid4()}.{file_extension}"

        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        with open(file_path, "wb") as buffer:
            buffer.write(image_bytes)

        image_url = upload_image_to_storage(
            image_bytes,
            unique_filename,
            file.content_type
        )

        return {
            "upload_id": unique_filename,
            "image_url": image_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_image(
    upload_id: str = Form(...),
    patient_name: str = Form(...),
    view_type: str = Form("frontal"),
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Runs AI analysis on uploaded image and saves complete case to Firestore.
    """

    try:

        file_path = os.path.join(UPLOAD_DIR, upload_id)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail="Uploaded file not found."
            )

        with open(file_path, "rb") as f:
            image_bytes = f.read()

        result = ai_engine.analyze_image(
            image_bytes,
            view_type=view_type
        )

        case_data = {
            "patient_name": patient_name,
            "image_url": result.get("details", {}).get("image_url", ""),
            "view_type": view_type,
            "status": "completed",
            "finishing_score": result.get("abo_score", 0),
            "alignment_score": result.get("arch_symmetry_score", 0),
            "confidence_score": result.get("confidence_score", 0),
            "midline_deviation_mm": result.get(
                "details", {}
            ).get(
                "overjet_overbite", {}
            ).get(
                "overjet_mm", 0
            ),
            "overjet_mm": result.get(
                "details", {}
            ).get(
                "overjet_overbite", {}
            ).get(
                "overjet_mm", 0
            ),
            "overbite_percent": result.get(
                "details", {}
            ).get(
                "overjet_overbite", {}
            ).get(
                "overbite_percent", 0
            ),
            "abo_score": result.get("abo_score", 0),
            "andrews_score": result.get("andrews_score", 0),
            "root_angulation_score": result.get(
                "root_angulation_score", 0
            ),
            "prediction": result.get("prediction", ""),
            "recommendations": result.get("recommendations", []),
            "metrics": result.get("details", {}),
            "created_at": None
        }

        saved_case = save_analysis_record(
            case_data,
            current_user.uid
        )

        return saved_case

    except Exception as e:

        print("\n========== AI ANALYSIS ERROR ==========")
        traceback.print_exc()
        print("=======================================\n")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/history", response_model=List[AnalysisRecordResponse])
async def get_history(
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Retrieves current user's analysis history.
    """

    try:
        history = get_user_analysis_history(current_user.uid)
        return history

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch history: {str(e)}"
        )


@router.get("/report/{record_id}", response_model=AnalysisRecordResponse)
async def get_analysis(
    record_id: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Retrieves a specific analysis record.
    """

    try:

        record = get_analysis_by_id(record_id)

        if not record:
            raise HTTPException(
                status_code=404,
                detail="Analysis record not found."
            )

        if record.get("user_id") != current_user.uid:
            raise HTTPException(
                status_code=403,
                detail="Access denied."
            )

        return record

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch record: {str(e)}"
        )