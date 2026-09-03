# app/core/detector.py
# PCB defect detection using YOLOv11.
#
# Key difference from SafeVision:
# SafeVision: detects presence/absence of PPE items
#             compliance = person has hardhat + vest
#
# PCB Defect: detects defect location and type
#             every detection IS a defect
#             no spatial association needed
#             result = list of defects found on PCB

import torch
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()

from app.config import settings

# Auto-detect device
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"PCBDetector using device: {DEVICE}")


@dataclass
class DefectDetection:
    """Single defect detection result."""
    defect_id:    int
    defect_type:  str
    confidence:   float
    bbox:         tuple          # (x1, y1, x2, y2) in pixels
    bbox_area_pct: float         # bbox area as % of image
    needs_verification: bool     # low confidence → send to GPT-4o
    verified_by_vlm: bool = False
    vlm_confirmed: Optional[bool] = None


@dataclass
class PCBAnalysis:
    """Complete analysis result for one PCB image."""
    image_path:      str
    total_defects:   int
    defect_types:    dict         # {defect_type: count}
    defections:      list[DefectDetection] = field(default_factory=list)
    needs_verification: int = 0
    quality_status:  str = "PASS"  # PASS, FAIL, UNCERTAIN

    @property
    def has_defects(self) -> bool:
        return self.total_defects > 0

    @property
    def defect_summary(self) -> str:
        if not self.has_defects:
            return "No defects detected — PCB passes inspection"
        parts = [f"{count} {dtype}" for dtype, count in self.defect_types.items()]
        return f"{self.total_defects} defects found: {', '.join(parts)}"


class PCBDetector:
    """
    YOLOv11-based PCB defect detector.

    Detects 6 defect types:
      missing_hole, mouse_bite, open_circuit,
      short, spur, spurious_copper

    Tiered confidence system:
      conf > verification_threshold → accept directly
      conf < verification_threshold → flag for GPT-4o
    """

    def __init__(self, model_path: str = None):
        model_path = model_path or settings.yolo_model_path

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run train.py first."
            )

        print(f"Loading PCB detector from {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = settings.class_names
        self.confidence_threshold = settings.confidence_threshold
        self.verification_threshold = settings.verification_threshold
        print(f"Detector ready. Classes: {self.class_names}")

    def analyze_pcb(self, image_path: str) -> PCBAnalysis:
        """
        Analyzes a PCB image for defects.

        Args:
            image_path: path to PCB image

        Returns:
            PCBAnalysis with all detected defects
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]

        # Run YOLO inference
        results = self.model(
            image_path,
            conf=self.confidence_threshold,
            iou=settings.iou_threshold,
            verbose=False,
            device=DEVICE,
        )

        defections = []
        defect_types = {}
        needs_verification = 0

        for result in results:
            if result.boxes is None:
                continue

            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                defect_type = self.class_names[class_id]

                # Bounding box in pixels
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_area = (x2 - x1) * (y2 - y1)
                bbox_area_pct = bbox_area / (w * h) * 100

                # Flag low confidence for GPT-4o verification
                needs_vlm = confidence < self.verification_threshold

                if needs_vlm:
                    needs_verification += 1

                detection = DefectDetection(
                    defect_id=i,
                    defect_type=defect_type,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    bbox_area_pct=bbox_area_pct,
                    needs_verification=needs_vlm,
                )
                defections.append(detection)

                # Count by type
                defect_types[defect_type] = (
                    defect_types.get(defect_type, 0) + 1
                )

        # Determine quality status
        if len(defections) == 0:
            quality_status = "PASS"
        elif needs_verification == len(defections):
            quality_status = "UNCERTAIN"
        else:
            quality_status = "FAIL"

        return PCBAnalysis(
            image_path=image_path,
            total_defects=len(defections),
            defect_types=defect_types,
            defections=defections,
            needs_verification=needs_verification,
            quality_status=quality_status,
        )

    def draw_results(
        self,
        image_path: str,
        analysis: PCBAnalysis,
        output_path: str = None
    ) -> np.ndarray:
        """
        Draws detection results on the image.

        Color coding:
          Red box:    confirmed defect (high confidence)
          Yellow box: uncertain (needs GPT-4o verification)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Color map per defect type
        colors = {
            "missing_hole":    (0, 0, 255),      # red
            "mouse_bite":      (0, 128, 255),     # orange
            "open_circuit":    (0, 255, 255),     # yellow
            "short":           (255, 0, 0),       # blue
            "spur":            (255, 0, 255),     # magenta
            "spurious_copper": (0, 255, 0),       # green
        }
        uncertain_color = (0, 255, 255)  # yellow for uncertain

        for detection in analysis.defections:
            x1, y1, x2, y2 = detection.bbox

            # Choose color
            if detection.needs_verification:
                color = uncertain_color
            else:
                color = colors.get(detection.defect_type, (0, 0, 255))

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = (
                f"{detection.defect_type} "
                f"{detection.confidence:.2f}"
                f"{' ?' if detection.needs_verification else ''}"
            )
            label_y = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(
                img, label, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        # Draw summary at top
        summary = (
            f"Defects: {analysis.total_defects} | "
            f"Status: {analysis.quality_status}"
        )
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            img, summary, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, img)

        return img