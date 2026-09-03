# app/api/routes/defects.py

from fastapi import APIRouter, Query
from pydantic import BaseModel
from app.core.database import query_defects, get_quality_summary

router = APIRouter()

class DefectRecord(BaseModel):
    id: int
    detected_at: str
    defect_type: str
    confidence: float
    bbox_area_pct: float
    needs_verification: bool
    verified_by_vlm: bool
    quality_status: str

class QualitySummary(BaseModel):
    total_inspections: int
    total_defects: int
    passed: int
    failed: int
    uncertain: int
    yield_rate: float
    avg_defects_per_pcb: float
    defects_by_type: dict

@router.get("/summary", response_model=QualitySummary)
async def get_summary(date_from: str = None):
    """Returns aggregate quality statistics."""
    summary = get_quality_summary(date_from=date_from)
    return QualitySummary(**summary)

@router.get("", response_model=list[DefectRecord])
async def list_defects(
    defect_type: str = Query(default=None),
    limit: int = Query(default=50, le=500),
    date_from: str = Query(default=None)
):
    """Returns defect records with optional filters."""
    defects = query_defects(
        defect_type=defect_type,
        date_from=date_from,
        limit=limit
    )
    return [
        DefectRecord(
            id=d["id"],
            detected_at=d["detected_at"],
            defect_type=d["defect_type"],
            confidence=d["confidence"],
            bbox_area_pct=d["bbox_area_pct"],
            needs_verification=bool(d["needs_verification"]),
            verified_by_vlm=bool(d["verified_by_vlm"]),
            quality_status=d["quality_status"],
        )
        for d in defects
    ]