# train.py
# Train YOLOv11 on PCB defect dataset with MLflow tracking.
#
# Key difference from SafeVision:
# PCB defects are extremely small (avg 0.09% of image area)
# vs PPE items which are much larger.
# This means we may need a larger model architecture
# and should monitor small object detection metrics carefully.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlflow
import torch
import shutil
import glob
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()

from app.config import settings

DB_PATH = Path(__file__).parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

def train(
    model_name: str = "yolo11n.pt",
    epochs: int = 50,
    batch: int = 32,
    imgsz: int = 640,
    run_label: str = None
):
    """
    Trains YOLOv11 on PCB defect dataset with MLflow tracking.

    Args:
        model_name: YOLO model architecture to use
        epochs:     number of training epochs
        batch:      batch size
        imgsz:      input image size
        run_label:  optional label for MLflow run name
    """
    # Find dataset
    yaml_files = list(Path("data/raw").rglob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError("data.yaml not found in data/raw")
    yaml_path = str(yaml_files[0].absolute())

    arch = model_name.replace(".pt", "")
    label = run_label or arch
    run_name = f"pcb_{label}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    print("=" * 60)
    print(f"PCB DEFECT DETECTION TRAINING")
    print("=" * 60)
    print(f"Model:    {model_name}")
    print(f"Epochs:   {epochs}")
    print(f"Batch:    {batch}")
    print(f"Dataset:  {yaml_path}")
    print(f"Run:      {run_name}")

    mlflow.set_experiment("pcb-defect-detection")

    with mlflow.start_run(run_name=run_name) as run:
        print(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_params({
            "model_architecture":   arch,
            "dataset":              "PCB-defect-roboflow",
            "num_classes":          6,
            "class_names":          ",".join(settings.class_names),
            "train_images":         3224,
            "val_images":           1592,
            "test_images":          537,
            "imgsz":                imgsz,
            "epochs":               epochs,
            "batch_size":           batch,
            "optimizer":            "AdamW",
            "lr0":                  0.001,
            "lrf":                  0.01,
            "avg_defect_bbox_pct":  0.09,
            "device":               "cuda" if torch.cuda.is_available() else "cpu",
        })

        model = YOLO(model_name)

        # Callback for real-time epoch logging
        def on_fit_epoch_end(trainer):
            epoch = trainer.epoch
            metrics = trainer.metrics
            log_dict = {}

            if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                loss_items = trainer.loss_items
                if hasattr(loss_items, "items"):
                    for key, val in loss_items.items():
                        log_dict[f"train_{key}"] = float(val)

            val_metric_map = {
                "val_precision":    "metrics/precision(B)",
                "val_recall":       "metrics/recall(B)",
                "val_mAP50":        "metrics/mAP50(B)",
                "val_mAP50_95":     "metrics/mAP50-95(B)",
                "val_box_loss":     "val/box_loss",
                "val_cls_loss":     "val/cls_loss",
                "val_dfl_loss":     "val/dfl_loss",
            }

            for log_key, metric_key in val_metric_map.items():
                if metric_key in metrics:
                    log_dict[log_key] = float(metrics[metric_key])

            if log_dict:
                mlflow.log_metrics(log_dict, step=epoch)
                print(
                    f"  Epoch {epoch}: "
                    f"mAP50={log_dict.get('val_mAP50', 0):.4f} logged"
                )

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        results = model.train(
            data=yaml_path,
            imgsz=imgsz,
            epochs=epochs,
            patience=15,
            batch=batch,
            workers=4,
            project="models",
            name=run_name,
            device=0 if torch.cuda.is_available() else "cpu",
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            cls=1.0,          # balanced classes — no weight adjustment needed
            mosaic=1.0,
            degrees=10.0,
            save=True,
            val=True,
            verbose=False,
        )

        # Final metrics
        map50 = results.results_dict.get("metrics/mAP50(B)", 0)
        map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0)
        precision = results.results_dict.get("metrics/precision(B)", 0)
        recall = results.results_dict.get("metrics/recall(B)", 0)

        mlflow.log_metrics({
            "final_val_mAP50":      map50,
            "final_val_mAP50_95":   map50_95,
            "final_val_precision":  precision,
            "final_val_recall":     recall,
        })

        # Find and save best model
        print("\nSearching for best model...")
        exact_path = (
            Path("runs") / "detect" / "models" /
            run_name / "weights" / "best.pt"
        )

        best = None
        if exact_path.exists():
            best = exact_path
        else:
            all_pts = glob.glob("**/best.pt", recursive=True)
            if all_pts:
                all_pts.sort(
                    key=lambda x: Path(x).stat().st_mtime,
                    reverse=True
                )
                best = Path(all_pts[0])

        if best and best.exists():
            Path("models/trained").mkdir(parents=True, exist_ok=True)
            dest = Path("models/trained/best_nano.pt")
            shutil.copy(str(best), str(dest))
            mlflow.log_artifact(str(best), artifact_path="model")
            print(f"Model saved: {dest} ({dest.stat().st_size/1e6:.1f} MB)")

        mlflow.set_tags({
            "deployment_ready":     str(map50 > 0.70),
            "framework":            "ultralytics",
            "final_mAP50":          f"{map50:.4f}",
            "domain":               "PCB defect detection",
        })

        print(f"\nTraining complete")
        print(f"Final mAP50:      {map50:.4f}")
        print(f"Final mAP50-95:   {map50_95:.4f}")
        print(f"Final Precision:  {precision:.4f}")
        print(f"Final Recall:     {recall:.4f}")
        print(f"View MLflow:      http://localhost:5000")

    return results

if __name__ == "__main__":
    train(
        model_name="yolo11n.pt",
        epochs=50,
        batch=32,
        imgsz=640,
    )