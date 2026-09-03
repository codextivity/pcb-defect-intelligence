# 🔍 PCB Defect Intelligence System

AI-powered PCB (Printed Circuit Board) defect detection combining YOLOv11 object
detection with a LangChain agent for natural language quality management queries.

**GitHub:** https://github.com/codextivity/pcb-defect-intelligence

---

## What It Does

Upload a PCB image and get instant defect analysis:

```
POST /inspect         → per-defect detection report with annotated image
POST /query           → natural language quality insights from inspection database
GET  /defects         → individual defect records with filters
GET  /defects/summary → aggregate quality statistics and yield rate
GET  /health          → service status and model load state
```

Ask questions in natural language:
```
"What is our yield rate today?"
"Which defect type is most common?"
"How many PCBs failed inspection?"
"What quality improvements do you recommend?"
```

---

## Detected Defect Types

| Defect | Description | Test mAP50 |
|---|---|---|
| missing_hole | Drill hole absent from PCB | 0.828 |
| short | Unintended copper connection | 0.852 |
| spurious_copper | Excess copper remaining | 0.833 |
| open_circuit | Broken conductor path | 0.745 |
| spur | Copper protrusion from trace | 0.718 |
| mouse_bite | Irregular edge damage | 0.657 |
| **Overall** | **All 6 defect classes** | **0.772** |

---

## System Architecture

```
PCB Image
    │
    ▼
┌─────────────────────┐
│   YOLOv11n          │  Detects 6 defect types
│   Defect Detector   │  val mAP50: 0.903
│   5.5MB model       │  test mAP50: 0.772
└────────┬────────────┘
         │
         ├── High confidence ──► Store in SQLite directly
         │
         └── Uncertain ─────────► Flag for GPT-4o verification
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │   SQLite DB  │
                                  │  Defect Log  │
                                  └──────┬───────┘
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │  LangChain  │
                                  │  Agent      │
                                  │  5 DB tools │
                                  └─────────────┘
                                         │
                                         ▼
                                  Natural Language
                                  Quality Reports
```

---

## Key Engineering Decisions

**Why YOLOv11 for PCB defect detection?**
PCB defects are extremely small — average bounding box area is 0.09% of the image.
YOLOv11 with imgsz=640 provides the resolution needed to detect these tiny defects
in real-time at 1.2ms inference speed.

**Why a tiered verification system?**
Mouse bite defects scored 0.657 mAP50 due to irregular shape and similarity to
normal PCB edges. Low-confidence detections are flagged for GPT-4o vision
verification — the same approach used in industrial inspection pipelines where
false positives are costly.

**Why LangChain agent over a fixed dashboard?**
Quality engineers ask unpredictable questions. A fixed dashboard only answers
predefined queries. The LangChain agent answers any quality question grounded in
real inspection data — defect trends, yield rates, and specific recommendations
per defect type.

**Why the dataset required no class weighting?**
The PCB defect dataset has exceptional balance — 1.2x imbalance ratio across
6 classes. This is unlike typical manufacturing datasets and enabled strong
performance across all defect types without special augmentation.

---

## Model Performance

Trained on 5,353 PCB images — YOLOv11n, 50 epochs, imgsz=640:

```
Dataset split:
  Train: 3,224 images
  Valid: 1,592 images
  Test:    537 images

Class balance: 1.2x imbalance ratio (excellent)
Avg defect size: 0.09% of image (very small)

Validation results:
  mAP50:      0.903
  mAP50-95:   0.503
  Precision:  0.948
  Recall:     0.795

Test results (held-out, never seen during training):
  mAP50:      0.772
  mAP50-95:   0.436
  Precision:  0.918
  Recall:     0.784
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Object detection | YOLOv11n | PCB defect detection |
| Vision verification | GPT-4o | Uncertain detection verification |
| LLM framework | LangChain + LangGraph | Agent with 5 database tools |
| Database | SQLite | Defect and inspection history |
| API | FastAPI | HTTP endpoints |
| Experiment tracking | MLflow | Training run comparison |
| Deployment | Docker | Containerization |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Service status |
| POST | /inspect | Upload PCB image for defect analysis |
| POST | /query | Natural language quality query |
| GET | /defects | List defect records with filters |
| GET | /defects/summary | Aggregate quality statistics |

---

## Quick Start

```bash
git clone https://github.com/codextivity/pcb-defect-intelligence
cd pcb-defect-intelligence
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs

---

## Configuration

```bash
# .env
OPENAI_API_KEY=your-openai-api-key
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=pcb-defect-intelligence
YOLO_MODEL_PATH=models/trained/best_nano.pt
CONFIDENCE_THRESHOLD=0.25
VERIFICATION_THRESHOLD=0.70
```

---

## Project Structure

```
pcb-defect-intelligence/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Typed settings via pydantic-settings
│   ├── api/routes/
│   │   ├── health.py            # GET /health
│   │   ├── inspect.py           # POST /inspect
│   │   ├── query.py             # POST /query
│   │   └── defects.py           # GET /defects
│   └── core/
│       ├── detector.py          # YOLOv11 inference + defect analysis
│       ├── agent.py             # LangGraph agent with 5 quality tools
│       └── database.py          # SQLite operations
│
├── models/
│   ├── trained/
│   │   └── best_nano.pt         # Trained YOLOv11n weights (5.5MB)
│   └── deployed/
│       └── best_nano.onnx       # ONNX export for CPU deployment
│
├── data/
│   └── eval_metrics.json        # Test set evaluation results
│
├── notebooks/
│   └── eda.py                   # Exploratory data analysis
│
├── train.py                     # Training with MLflow tracking
├── evaluate.py                  # Test set evaluation
├── generate_traffic.py          # API load testing
└── params.yaml                  # Centralized training parameters
```

---

## Deployment Notes

The system requires approximately 300-400MB RAM for the ONNX model
alongside the LangChain agent.

```bash
# Local deployment
uvicorn app.main:app --reload --port 8000

# Docker
docker-compose up
```

---

## Interview Talking Points

**On the detection architecture:**
"PCB defects are extremely small — average bounding box area is 0.09%
of the image, compared to 5-9% for PPE items in my SafeVision project.
This required careful threshold tuning and a higher precision model.
The YOLOv11n achieved 0.903 mAP50 on validation and 0.772 on the
held-out test set across all 6 defect classes."

**On the dataset quality:**
"The PCB dataset had exceptional class balance at 1.2x imbalance ratio
across 6 classes. This is rare in manufacturing datasets and directly
contributed to strong performance across all defect types without
requiring special class weighting or augmentation strategies."

**On the tiered verification:**
"Mouse bite defects scored 0.657 mAP50 due to their irregular shape
and visual similarity to normal PCB edges. Rather than accepting this
limitation, I designed a tiered system where low-confidence detections
are flagged for GPT-4o vision verification — reducing false positives
in the quality record without missing real defects."

**On the LangChain agent:**
"Quality engineers do not want to write SQL queries. The LangChain agent
translates natural language questions into database queries and returns
actionable insights. It generates defect-specific recommendations —
for missing holes it suggests checking drill bit condition, for open
circuits it recommends reviewing etching time. These recommendations
are grounded in real inspection data, not generic advice."

**On relevance to 삼성전기:**
"This project directly maps to Samsung Electro-Mechanics' MLCC and
substrate inspection requirements. The same architecture — YOLO detection,
confidence-based routing, GPT-4o verification, and natural language
quality reporting — applies to any manufacturing inspection domain.
The transition from PCB defects to MLCC defects requires only
retraining on the target dataset."

---

## Version 2 Roadmap

- Anomaly detection layer — MVTec AD dataset for unseen defect types
- TensorRT export — optimize for NVIDIA Jetson edge deployment
- Multi-PCB batch processing — analyze entire production batch at once
- Automated retraining pipeline — triggered by drift detection
- Prometheus + Grafana monitoring — production metrics dashboard
- Kubernetes deployment — horizontal scaling for high-volume lines

---

## Author

Built by [Codextivity](https://github.com/codextivity) applying
Computer Vision expertise to electronics manufacturing inspection.

**Related projects:**
- [SafeVision PPE Compliance System](https://github.com/codextivity/safevision)
- [LangChain Research Copilot](https://github.com/codextivity/langchain-copilot)
- [Multimodal Document Intelligence](https://github.com/codextivity/multimodal-doc-intelligence)
