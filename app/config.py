# app/config.py

from pydantic_settings import BaseSettings
from pathlib import Path

ENV_FILE_PATH = Path(__file__).parent.parent / ".env"

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = ""
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "pcb-defect-intelligence"
    roboflow_api_key: str = ""

    # Model settings
    openai_chat_model: str = "gpt-4o-mini"
    yolo_model_path: str = "models/trained/best_nano.pt"
    yolo_base_model: str = "yolo11n.pt"

    # Class names — 6 PCB defect types
    class_names: list[str] = [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper"
    ]

    defect_classes: list[str] = [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper"
    ]

    # Training settings
    imgsz: int = 640
    epochs: int = 50
    batch_size: int = 32
    workers: int = 4

    # Inference settings
    confidence_threshold: float = 0.25
    verification_threshold: float = 0.70
    iou_threshold: float = 0.45

    # Storage
    database_path: str = "data/defects.db"

    model_config = {
        "env_file": str(ENV_FILE_PATH),
        "extra": "ignore",
        "case_sensitive": False,
    }

settings = Settings()