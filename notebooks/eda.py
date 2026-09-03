# notebooks/eda.py
# Exploratory Data Analysis for PCB Defect Dataset
#
# Why EDA before training?
# Understanding class distribution prevents surprises.
# If one defect class has 10x fewer images than others,
# we need to handle class imbalance before training.
# This is exactly what happened in SafeVision where
# violation classes were underrepresented.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import cv2
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_eda(data_root: str = "data/raw"):
    data_root = Path(data_root)

    # Find data.yaml
    yaml_files = list(data_root.rglob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"data.yaml not found in {data_root}")

    yaml_path = yaml_files[0]
    print(f"Dataset config: {yaml_path}")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    class_names = config["names"]
    num_classes = config["nc"]

    print(f"\nDataset Overview:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {class_names}")

    # Count images and annotations per split
    splits = ["train", "valid", "test"]
    split_stats = {}

    total_class_counts = defaultdict(int)
    total_images = 0
    total_labels = 0

    for split in splits:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"

        if not img_dir.exists():
            # Try alternative naming
            img_dir = data_root / split / "images"
            if not img_dir.exists():
                print(f"  {split}: directory not found — skipping")
                continue

        images = (
            list(img_dir.glob("*.jpg")) +
            list(img_dir.glob("*.jpeg")) +
            list(img_dir.glob("*.png"))
        )

        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        # Count class instances
        class_counts = defaultdict(int)
        bbox_areas = []
        images_with_defects = 0
        images_without_defects = 0

        for lbl_path in labels:
            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]

            if lines:
                images_with_defects += 1
                for line in lines:
                    parts = line.split()
                    class_id = int(parts[0])
                    bw = float(parts[3])
                    bh = float(parts[4])
                    area = bw * bh * 100
                    class_counts[class_names[class_id]] += 1
                    total_class_counts[class_names[class_id]] += 1
                    bbox_areas.append(area)
            else:
                images_without_defects += 1

        split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "class_counts": dict(class_counts),
            "images_with_defects": images_with_defects,
            "images_without_defects": images_without_defects,
            "avg_bbox_area": np.mean(bbox_areas) if bbox_areas else 0,
        }

        total_images += len(images)
        total_labels += len(labels)

        print(f"\n  {split.upper()} split:")
        print(f"    Images:              {len(images)}")
        print(f"    Labels:              {len(labels)}")
        print(f"    With defects:        {images_with_defects}")
        print(f"    Without defects:     {images_without_defects}")
        print(f"    Avg bbox area:       {np.mean(bbox_areas):.3f}% of image" if bbox_areas else "    No bboxes found")
        print(f"    Class distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"      {cls:25} {count:5d}")

    # Overall summary
    print(f"\n{'='*60}")
    print(f"OVERALL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total images:    {total_images}")
    print(f"Total labels:    {total_labels}")
    print(f"\nClass distribution (all splits):")
    print(f"{'Class':25} {'Count':>8} {'%':>8}")
    print(f"{'-'*45}")

    total_instances = sum(total_class_counts.values())
    for cls in class_names:
        count = total_class_counts.get(cls, 0)
        pct = count / total_instances * 100 if total_instances > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{cls:25} {count:>8} {pct:>7.1f}% {bar}")

    # Class imbalance check
    if total_class_counts:
        max_class = max(total_class_counts, key=total_class_counts.get)
        min_class = min(total_class_counts, key=total_class_counts.get)
        imbalance_ratio = (
            total_class_counts[max_class] /
            total_class_counts[min_class]
            if total_class_counts[min_class] > 0 else float('inf')
        )

        print(f"\nClass Imbalance Analysis:")
        print(f"  Most common:  {max_class} ({total_class_counts[max_class]})")
        print(f"  Least common: {min_class} ({total_class_counts[min_class]})")
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}x")

        if imbalance_ratio > 3:
            print(f"  ⚠ Significant imbalance detected")
            print(f"    → Consider cls_loss_weight adjustment in training")
            print(f"    → Consider data augmentation for minority classes")
        else:
            print(f"  ✅ Class distribution is relatively balanced")

    # Save class distribution chart
    Path("data").mkdir(exist_ok=True)
    if total_class_counts:
        fig, ax = plt.subplots(figsize=(10, 5))
        classes = list(total_class_counts.keys())
        counts = [total_class_counts[c] for c in classes]
        colors = ['#003580'] * len(classes)
        bars = ax.bar(classes, counts, color=colors)
        ax.set_title('PCB Defect Class Distribution', fontsize=14)
        ax.set_xlabel('Defect Type')
        ax.set_ylabel('Number of Instances')
        ax.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.5,
                str(count),
                ha='center', va='bottom', fontsize=10
            )
        plt.tight_layout()
        plt.savefig('data/class_distribution.png', dpi=150)
        print(f"\nChart saved: data/class_distribution.png")

    # Sample image analysis
    print(f"\nSample Image Analysis:")
    train_img_dir = data_root / "train" / "images"
    if train_img_dir.exists():
        sample_images = list(train_img_dir.glob("*.jpg"))[:5]
        for img_path in sample_images:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                print(f"  {img_path.name}: {w}x{h}")

    print(f"\n{'='*60}")
    print(f"EDA Complete")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Review class distribution chart: data/class_distribution.png")
    print(f"  2. Adjust cls_loss_weight if imbalance > 3x")
    print(f"  3. Run training: python train.py")

    return split_stats

if __name__ == "__main__":
    run_eda()