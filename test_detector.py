# test_detector.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").absolute()))

from dotenv import load_dotenv
load_dotenv()

from app.core.detector import PCBDetector

detector = PCBDetector("models/trained/best_nano.pt")

# Test on sample images
test_images = (
    list(Path("data/raw/test/images").glob("*.jpg")) +
    list(Path("data/raw/test/images").glob("*.png"))
)

print(f"Found {len(test_images)} test images")
print("Scanning for images with defects...\n")

Path("data/samples").mkdir(exist_ok=True)

found = 0
for img_path in test_images[:30]:
    analysis = detector.analyze_pcb(str(img_path))

    if analysis.has_defects:
        found += 1
        print(f"✅ {img_path.name}")
        print(f"   Defects: {analysis.total_defects}")
        print(f"   Types:   {analysis.defect_types}")
        print(f"   Status:  {analysis.quality_status}")
        print(f"   Summary: {analysis.defect_summary}")

        # Save annotated image for first defect found
        if found == 1:
            output = f"data/samples/defect_{img_path.stem}.jpg"
            detector.draw_results(str(img_path), analysis, output)
            print(f"   Saved:   {output}")
        print()

    if found >= 5:
        break

print(f"\nFound {found} images with defects in first 30 tested")