# app/main.py

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path

from app.api.routes import health, inspect, query, defects
from app.config import settings
from app.core.database import initialize_database

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting PCBVision API...")
    initialize_database()

    app.state.detector = None
    app.state.agent = None

    import asyncio

    async def warmup():
        await asyncio.sleep(3)
        try:
            from app.core.detector import PCBDetector
            from app.core.agent import build_quality_agent
            from app.config import settings
            from pathlib import Path

            if Path(settings.yolo_model_path).exists():
                print(f"Loading detector: {settings.yolo_model_path}")
                app.state.detector = PCBDetector(settings.yolo_model_path)
                print("Detector ready")

            print("Building quality agent...")
            app.state.agent = build_quality_agent()
            print("Agent ready")

        except Exception as e:
            print(f"Warmup failed: {e}")

    asyncio.create_task(warmup())
    yield
    print("Shutting down PCBVision API...")

app = FastAPI(
    title="PCBVision — PCB Defect Intelligence API",
    description=(
        "AI-powered PCB defect detection combining YOLOv11 "
        "inspection with LangChain agent for natural language "
        "quality management queries."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(inspect.router, prefix="/inspect", tags=["Inspection"])
app.include_router(query.router, prefix="/query", tags=["Agent"])
app.include_router(defects.router, prefix="/defects", tags=["Defects"])