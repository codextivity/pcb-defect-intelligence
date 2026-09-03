# app/core/database.py
# SQLite database for PCB defect history.
#
# Tracks every defect detected across all inspected PCBs.
# The LangChain agent queries this database to answer
# quality management questions in natural language.

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()

from app.config import settings


@contextmanager
def get_connection():
    """Context manager for SQLite connections."""
    db_path = Path(settings.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database():
    """
    Creates database tables if they do not exist.

    Two tables:
      inspections: one row per PCB image analyzed
      defects:     one row per defect detected
    """
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                analyzed_at     TEXT NOT NULL,
                image_path      TEXT NOT NULL,
                total_defects   INTEGER NOT NULL DEFAULT 0,
                quality_status  TEXT NOT NULL DEFAULT 'PASS',
                defect_types    TEXT NOT NULL DEFAULT '{}',
                needs_verification INTEGER NOT NULL DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS defects (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id   INTEGER NOT NULL,
                detected_at     TEXT NOT NULL,
                defect_type     TEXT NOT NULL,
                confidence      REAL NOT NULL,
                bbox_x1         INTEGER NOT NULL,
                bbox_y1         INTEGER NOT NULL,
                bbox_x2         INTEGER NOT NULL,
                bbox_y2         INTEGER NOT NULL,
                bbox_area_pct   REAL NOT NULL,
                needs_verification INTEGER NOT NULL DEFAULT 0,
                verified_by_vlm INTEGER NOT NULL DEFAULT 0,
                vlm_confirmed   INTEGER,
                FOREIGN KEY (inspection_id) REFERENCES inspections(id)
            )
        """)

    print(f"Database initialized at {settings.database_path}")


def store_pcb_analysis(analysis) -> int:
    """
    Stores a PCBAnalysis result in the database.

    Returns the inspection ID for reference.
    """
    with get_connection() as conn:
        # Store inspection summary
        cursor = conn.execute("""
            INSERT INTO inspections (
                analyzed_at, image_path, total_defects,
                quality_status, defect_types, needs_verification
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            analysis.image_path,
            analysis.total_defects,
            analysis.quality_status,
            json.dumps(analysis.defect_types),
            analysis.needs_verification,
        ))

        inspection_id = cursor.lastrowid

        # Store individual defects
        for detection in analysis.defections:
            x1, y1, x2, y2 = detection.bbox
            conn.execute("""
                INSERT INTO defects (
                    inspection_id, detected_at, defect_type,
                    confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    bbox_area_pct, needs_verification,
                    verified_by_vlm, vlm_confirmed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inspection_id,
                datetime.now().isoformat(),
                detection.defect_type,
                detection.confidence,
                x1, y1, x2, y2,
                detection.bbox_area_pct,
                int(detection.needs_verification),
                int(detection.verified_by_vlm),
                detection.vlm_confirmed,
            ))

    return inspection_id


def get_quality_summary(date_from: str = None) -> dict:
    """
    Returns aggregate quality statistics.
    Used by LangChain agent to answer quality questions.
    """
    where = f"WHERE analyzed_at >= '{date_from}'" if date_from else ""

    with get_connection() as conn:
        # Inspection summary
        stats = conn.execute(f"""
            SELECT
                COUNT(*) as total_inspections,
                SUM(total_defects) as total_defects,
                SUM(CASE WHEN quality_status='PASS' THEN 1 ELSE 0 END)
                    as passed,
                SUM(CASE WHEN quality_status='FAIL' THEN 1 ELSE 0 END)
                    as failed,
                SUM(CASE WHEN quality_status='UNCERTAIN' THEN 1 ELSE 0 END)
                    as uncertain,
                AVG(total_defects) as avg_defects_per_pcb
            FROM inspections {where}
        """).fetchone()

        # Defect type breakdown
        defect_counts = conn.execute(f"""
            SELECT d.defect_type, COUNT(*) as count
            FROM defects d
            JOIN inspections i ON d.inspection_id = i.id
            {where.replace('WHERE', 'WHERE i.')}
            GROUP BY d.defect_type
            ORDER BY count DESC
        """).fetchall()

    total = stats["total_inspections"] or 0
    passed = stats["passed"] or 0
    yield_rate = passed / total if total > 0 else 1.0

    return {
        "total_inspections":    total,
        "total_defects":        stats["total_defects"] or 0,
        "passed":               passed,
        "failed":               stats["failed"] or 0,
        "uncertain":            stats["uncertain"] or 0,
        "yield_rate":           yield_rate,
        "avg_defects_per_pcb":  stats["avg_defects_per_pcb"] or 0,
        "defects_by_type": {
            row["defect_type"]: row["count"]
            for row in defect_counts
        }
    }


def query_defects(
    defect_type: str = None,
    date_from: str = None,
    limit: int = 50
) -> list[dict]:
    """Returns individual defect records with optional filters."""
    conditions = []
    if defect_type:
        conditions.append(f"d.defect_type = '{defect_type}'")
    if date_from:
        conditions.append(f"d.detected_at >= '{date_from}'")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_connection() as conn:
        rows = conn.execute(f"""
            SELECT
                d.id, d.detected_at, d.defect_type,
                d.confidence, d.bbox_area_pct,
                d.needs_verification, d.verified_by_vlm,
                i.quality_status, i.image_path
            FROM defects d
            JOIN inspections i ON d.inspection_id = i.id
            {where}
            ORDER BY d.detected_at DESC
            LIMIT {limit}
        """).fetchall()

    return [dict(row) for row in rows]