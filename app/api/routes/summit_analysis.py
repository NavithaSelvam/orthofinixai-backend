import json
import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session

from app.api.deps_summit import get_current_summit_user
from app.db.orm_models import User, AnalysisReport
from app.db.sqlalchemy_db import get_db_session
from app.db.firebase import UPLOADS_DIR, upload_image_to_storage
from app.models.summit_schemas import (
    UploadResponse,
    AnalyzeRequest,
    AnalysisReportResponse,
    HistoryItem,
)
from app.services.report_builder import (
    build_report_from_ai,
    report_to_response,
)

router = APIRouter()

# In-memory upload staging (upload_id -> path)
_upload_cache: dict[str, dict] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_summit_user),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    data = await file.read()
    ext = file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg"
    upload_id = str(uuid.uuid4())
    filename = f"{upload_id}.{ext}"
    filepath = os.path.join(UPLOADS_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(data)
    image_url = upload_image_to_storage(data, filename, file.content_type or "image/jpeg")
    _upload_cache[upload_id] = {"path": filepath, "bytes": data, "url": image_url}
    return UploadResponse(upload_id=upload_id, image_url=image_url, filename=filename)


@router.post("/analyze", response_model=AnalysisReportResponse)
async def analyze(
    upload_id: str = Form(None),
    patient_name: str = Form("Patient"),
    view_type: str = Form("frontal"),
    file: UploadFile = File(None),
    current_user: User = Depends(get_current_summit_user),
    db: Session = Depends(get_db_session),
):
    image_bytes = None
    image_url = None

    if upload_id and upload_id in _upload_cache:
        cached = _upload_cache[upload_id]
        image_bytes = cached["bytes"]
        image_url = cached["url"]
    elif file and file.filename:
        image_bytes = await file.read()
        image_url = upload_image_to_storage(
            image_bytes, file.filename, file.content_type or "image/jpeg"
        )
    else:
        raise HTTPException(400, "Provide upload_id or image file")

    try:
        report = build_report_from_ai(
            db, current_user.id, image_bytes, patient_name, image_url, view_type
        )
    except ValueError as e:
        if upload_id in _upload_cache:
            del _upload_cache[upload_id]
        raise HTTPException(status_code=400, detail=str(e))
        
    if upload_id in _upload_cache:
        del _upload_cache[upload_id]
    return report_to_response(report)


@router.get("/history", response_model=List[HistoryItem])
def history(
    current_user: User = Depends(get_current_summit_user),
    db: Session = Depends(get_db_session),
):
    rows = (
        db.query(AnalysisReport)
        .filter(AnalysisReport.user_id == current_user.id)
        .order_by(AnalysisReport.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        HistoryItem(
            id=r.id,
            patient_name=r.patient_name,
            finishing_score=r.finishing_score,
            confidence_score=r.confidence_score,
            created_at=r.created_at,
            image_url=r.image_url,
        )
        for r in rows
    ]


@router.get("/report/{report_id}", response_model=AnalysisReportResponse)
def get_report(
    report_id: str,
    current_user: User = Depends(get_current_summit_user),
    db: Session = Depends(get_db_session),
):
    report = (
        db.query(AnalysisReport)
        .filter(AnalysisReport.id == report_id, AnalysisReport.user_id == current_user.id)
        .first()
    )
    if not report:
        raise HTTPException(404, "Report not found")
    return report_to_response(report)
