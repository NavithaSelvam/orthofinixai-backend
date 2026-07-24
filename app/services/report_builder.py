"""Maps AI engine output to STAR Summit report format."""
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.db.orm_models import AnalysisReport
from app.services.ai_engine import ai_engine


def _extract_metrics(result: dict) -> Dict[str, Any]:
    details = result.get("details") or {}
    lateral = details.get("overjet_overbite") or {}
    andrews = details.get("andrews_details") or {}

    # Find midline discrepancy in andrews details (it's a list)
    midline = 0.0
    if isinstance(andrews, list):
        for key_data in andrews:
            if isinstance(key_data, dict) and "midline" in key_data.get("key", "").lower():
                midline = key_data.get("deviation_mm", 0.0)

    overjet = lateral.get("overjet_mm", lateral.get("overjet", 0.0))
    overbite = lateral.get("overbite_percent", lateral.get("overbite", 0.0))

    finishing = (
        result.get("andrews_score", 0) * 0.35
        + result.get("arch_symmetry_score", 0) * 0.25
        + result.get("root_angulation_score", 0) * 0.2
        + (100 - min(result.get("abo_score", 0), 40)) * 0.2
    )

    molar_left = "Class I"
    molar_right = "Class I"
    if isinstance(andrews, list):
        molar_details = next((k for k in andrews if "Molar" in k.get("key", "")), {})
        molar_left = molar_details.get("details", {}).get("left", {}).get("classification", "Class I")
        molar_right = molar_details.get("details", {}).get("right", {}).get("classification", "Class I")


    return {
        "midline_deviation_mm": round(float(midline), 1),
        "overjet_mm": round(float(overjet), 1),
        "overbite_percent": round(float(overbite), 1),
        "finishing_score": round(finishing, 1),
        "alignment_score": round(result.get("arch_symmetry_score", 0.0), 1),
        "molar_right": molar_right,
        "molar_left": molar_left,
        "occlusion_summary": lateral.get("summary", ""),
        "warnings": details.get("warnings", []),
        "conflicts": details.get("conflicts", []),
    }


def build_report_from_ai(
    db: Session,
    user_id: str,
    image_bytes: bytes,
    patient_name: str,
    image_url: str,
    view_type: str = "frontal",
) -> AnalysisReport:
    result = ai_engine.analyze_image(image_bytes, view_type=view_type)
    metrics = _extract_metrics(result)
    report_id = str(uuid.uuid4())

    report = AnalysisReport(
        id=report_id,
        user_id=user_id,
        patient_name=patient_name,
        image_url=image_url,
        view_type=view_type,
        status="completed",
        finishing_score=metrics["finishing_score"],
        alignment_score=metrics["alignment_score"],
        confidence_score=float(result.get("confidence_score", 0.0)) * 100,
        midline_deviation_mm=metrics["midline_deviation_mm"],
        overjet_mm=metrics["overjet_mm"],
        overbite_percent=metrics["overbite_percent"],
        abo_score=float(result.get("abo_score", 0.0)),
        andrews_score=float(result.get("andrews_score", 0.0)),
        prediction=result.get("prediction", "Analysis complete."),
        recommendations_json=json.dumps(result.get("recommendations", [])),
        metrics_json=json.dumps({**metrics, "details": result.get("details", {})}),
        created_at=datetime.now(timezone.utc),
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def report_to_response(report: AnalysisReport):
    from app.models.summit_schemas import AnalysisReportResponse
    import json as _json

    return AnalysisReportResponse(
        id=report.id,
        patient_name=report.patient_name,
        image_url=report.image_url,
        view_type=report.view_type,
        status=report.status,
        finishing_score=report.finishing_score,
        alignment_score=report.alignment_score,
        confidence_score=report.confidence_score,
        midline_deviation_mm=report.midline_deviation_mm,
        overjet_mm=report.overjet_mm,
        overbite_percent=report.overbite_percent,
        abo_score=report.abo_score,
        andrews_score=report.andrews_score,
        prediction=report.prediction,
        recommendations=_json.loads(report.recommendations_json or "[]"),
        metrics=_json.loads(report.metrics_json or "{}"),
        created_at=report.created_at,
    )
