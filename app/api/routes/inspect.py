# app/api/routes/inspect.py

import tempfile
import os
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from pathlib import Path
from app.core.database import store_pcb_analysis

router = APIRouter()

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

class DefectResult(BaseModel):
    defect_id:          int
    defect_type:        str
    confidence:         float
    bbox:               list[int]
    bbox_area_pct:      float
    needs_verification: bool

class InspectionResponse(BaseModel):
    inspection_id:      int
    total_defects:      int
    quality_status:     str
    defect_types:       dict
    defect_summary:     str
    defects:            list[DefectResult]
    annotated_image_base64: str = ""

@router.post("", response_model=InspectionResponse)
async def inspect_pcb(
    request: Request,
    file: UploadFile = File(...),
    return_image: bool = True
):
    """
    Upload a PCB image for defect inspection.

    Returns detected defects with type, location,
    and confidence score. Flags uncertain detections
    for GPT-4o verification.
    """
    if request.app.state.detector is None:
        from app.core.detector import PCBDetector
        from app.config import settings

        model_path = settings.yolo_model_path
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {model_path}"
            )
        request.app.state.detector = PCBDetector(model_path)

    file_ext = "." + file.filename.split(".")[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}"
        )

    detector = request.app.state.detector

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=file_ext
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        analysis = detector.analyze_pcb(tmp_path)
        inspection_id = store_pcb_analysis(analysis)

        defects = [
            DefectResult(
                defect_id=d.defect_id,
                defect_type=d.defect_type,
                confidence=d.confidence,
                bbox=list(d.bbox),
                bbox_area_pct=d.bbox_area_pct,
                needs_verification=d.needs_verification,
            )
            for d in analysis.defections
        ]

        annotated_b64 = ""
        if return_image:
            import cv2
            annotated = detector.draw_results(tmp_path, analysis)
            _, buffer = cv2.imencode(".jpg", annotated)
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        return InspectionResponse(
            inspection_id=inspection_id,
            total_defects=analysis.total_defects,
            quality_status=analysis.quality_status,
            defect_types=analysis.defect_types,
            defect_summary=analysis.defect_summary,
            defects=defects,
            annotated_image_base64=annotated_b64,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(tmp_path)