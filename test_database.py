# test_database.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").absolute()))

from dotenv import load_dotenv
load_dotenv()

from app.core.detector import PCBDetector
from app.core.database import (
    initialize_database,
    store_pcb_analysis,
    get_quality_summary
)

initialize_database()
detector = PCBDetector("models/trained/best_nano.pt")

test_images = list(Path("data/raw/test/images").glob("*.jpg"))[:20]

print(f"Running batch inspection on 20 images...\n")

for img_path in test_images:
    analysis = detector.analyze_pcb(str(img_path))
    inspection_id = store_pcb_analysis(analysis)

    status_icon = {
        "PASS": "✅",
        "FAIL": "❌",
        "UNCERTAIN": "⚠"
    }.get(analysis.quality_status, "?")

    print(
        f"{status_icon} [{inspection_id:3d}] "
        f"{img_path.name[:40]:40} "
        f"Defects: {analysis.total_defects}"
    )

print("\n" + "="*60)
print("QUALITY SUMMARY")
print("="*60)

summary = get_quality_summary()
total = summary["total_inspections"]

print(f"Total inspections:     {total}")
print(f"Passed:                {summary['passed']}")
print(f"Failed:                {summary['failed']}")
print(f"Uncertain:             {summary['uncertain']}")
print(f"Yield rate:            {summary['yield_rate']:.1%}")
print(f"Total defects found:   {summary['total_defects']}")
print(f"Avg defects per PCB:   {summary['avg_defects_per_pcb']:.2f}")
print(f"\nDefects by type:")
for dtype, count in summary["defects_by_type"].items():
    print(f"  {dtype:25} {count:5d}")