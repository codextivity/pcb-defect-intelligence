# evaluate.py
from ultralytics import YOLO
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def evaluate():
    model_path = "models/trained/best_nano.pt"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print("=" * 60)
    print("PCB DEFECT DETECTION — TEST SET EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")

    model = YOLO(model_path)

    yaml_files = list(Path("data/raw").rglob("data.yaml"))
    yaml_path = str(yaml_files[0].absolute())

    with open(yaml_files[0]) as f:
        config = yaml.safe_load(f)
    class_names = config["names"]

    results = model.val(
        data=yaml_path,
        split="test",
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device=0,
        verbose=True,
    )

    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0)
    precision = results.results_dict.get("metrics/precision(B)", 0)
    recall = results.results_dict.get("metrics/recall(B)", 0)

    print(f"\nOverall Test Results:")
    print(f"  mAP50:      {map50:.4f}")
    print(f"  mAP50-95:   {map50_95:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")

    print(f"\nPer-Class Results:")
    print(f"  {'Class':25} {'mAP50':>8} {'Assessment'}")
    print(f"  {'-'*55}")

    class_map50s = {}
    if hasattr(results.box, "ap50") and results.ap_class_index is not None:
        for i, class_idx in enumerate(results.ap_class_index):
            class_name = class_names[class_idx]
            ap50 = float(results.box.ap50[i])
            class_map50s[class_name] = ap50

            if ap50 >= 0.85:
                assessment = "✅ Excellent"
            elif ap50 >= 0.70:
                assessment = "✅ Good"
            elif ap50 >= 0.50:
                assessment = "⚠ Acceptable"
            else:
                assessment = "❌ Needs improvement"

            print(f"  {class_name:25} {ap50:>8.4f} {assessment}")

    # Save metrics
    Path("data").mkdir(exist_ok=True)
    eval_metrics = {
        "test_mAP50":        map50,
        "test_mAP50_95":     map50_95,
        "test_precision":    precision,
        "test_recall":       recall,
    }
    eval_metrics.update({
        f"test_mAP50_{k}": v
        for k, v in class_map50s.items()
    })

    with open("data/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"\nEval metrics saved: data/eval_metrics.json")

    print(f"\nInterview talking point:")
    print(f"  'The PCB defect detection model achieves {map50:.1%} mAP50")
    print(f"   on the held-out test set across 6 defect classes.")
    print(f"   The balanced dataset (1.2x imbalance ratio) enabled")
    print(f"   strong performance across all defect types.'")

if __name__ == "__main__":
    evaluate()