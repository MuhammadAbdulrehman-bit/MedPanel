# MediScan AI 🩺
### Autonomous Medical Imaging Triage System

> A production-grade multi-agent AI system that analyzes medical images and generates structured clinical reports. Upload a chest X-ray, skin lesion, or retinal scan — three specialized AI agents analyze the image, retrieve relevant medical literature, and synthesize a reasoning-first clinical report.

**[Live Demo](https://your-vercel-url.vercel.app)** · **[API](https://fhkhthjkkl-medpanel.hf.space)** · **[HuggingFace Space](https://huggingface.co/spaces/FHKHTHJKKL/MedPanel)**

---

## What Makes This Different

Most medical AI projects wrap a model's output in a template and call it a report. MediScan does something different — the Report Agent **reasons**.

When a CV model says "Melanocytic Nevus (98% confidence)" but the patient's lesion is on their lower lip, the agent flags it:

> *"While the visual model suggests Melanocytic Nevus, the clinical location of the lower lip necessitates a differential diagnosis of Actinic Keratosis or Basal Cell Carcinoma — both statistically dominant at this site and warranting urgent dermatology evaluation."*

The system challenges its own CV predictions using medical literature. That's the point.

---

## Architecture

```
User Input (image + patient context + body location)
                    │
        ┌───────────▼───────────┐
        │   Agent 1: Vision     │
        │   Triage Agent        │
        │                       │
        │  Routes to correct    │
        │  CV model based on    │
        │  scan type            │
        │                       │
        │  • EfficientNetB3     │  ← Skin Lesion (HAM10000, 92% val acc)
        │  • ResNet50           │  ← Diabetic Retinopathy (81% val acc)
        │  • DenseNet121        │  ← Chest X-Ray (TorchXRayVision)
        └───────────┬───────────┘
                    │ CV findings + confidence scores
        ┌───────────▼───────────┐
        │   Agent 2: RAG        │
        │   Literature Agent    │
        │                       │
        │  Semantic search on   │
        │  260-chunk medical    │
        │  knowledge base       │
        │  (ChromaDB)           │
        │                       │
        │  Location-aware:      │
        │  "skin on lower lip"  │
        │  → retrieves BCC/AK   │
        │    context            │
        └───────────┬───────────┘
                    │ Medical literature context
        ┌───────────▼───────────┐
        │   Agent 3: Report     │
        │   Synthesis Agent     │
        │                       │
        │  Groq (Llama 3.1 8B) │
        │                       │
        │  Reconciliation:      │
        │  CV findings vs       │
        │  clinical literature  │
        │  vs patient context   │
        │                       │
        │  Flags malignancy     │
        │  risks. Honest about  │
        │  uncertainty.         │
        └───────────┬───────────┘
                    │
        Structured Clinical Report
```

---

## CV Models

| Model | Task | Dataset | Val Accuracy | Classes |
|-------|------|---------|-------------|---------|
| EfficientNetB3 | Skin Lesion Classification | HAM10000 (10,015 images) | **92%** | 7 |
| ResNet50 | Diabetic Retinopathy Grading | DR Gaussian 224×224 | **81%** | 5 |
| DenseNet121 | Chest X-Ray Pathology | NIH + CheXpert + MIMIC (TorchXRayVision) | Pretrained | 18 |

All models use transfer learning. Skin and retinal models trained from scratch on Kaggle (free P100 GPU) with oversampling to handle class imbalance.

---

## RAG Knowledge Base

260 chunks across 20 disease files indexed in ChromaDB using `sentence-transformers` (all-MiniLM-L6-v2).

Each disease file contains:
- Visual appearance on scan
- Typical body locations
- Differential diagnosis (most important — used for reasoning)
- Location-based clues
- Clinical signs and risk factors
- Common diagnostic mistakes

**Collections:**
- `medical_xray` — 104 chunks (pneumonia, effusion, cardiomegaly, atelectasis, pneumothorax, pulmonary edema, emphysema, consolidation)
- `medical_skin` — 92 chunks (melanoma, BCC, actinic keratosis, melanocytic nevus, dermatofibroma, vascular lesion, benign keratosis)
- `medical_retinal` — 64 chunks (No DR through Proliferative DR)

---

## Tech Stack

**Backend**
- `FastAPI` — REST API
- `LangGraph` — Agent state machine orchestration
- `Groq API` — Llama 3.1 8B for report synthesis (free tier)
- `ChromaDB` — Vector database for RAG
- `sentence-transformers` — Embedding model
- `PyTorch` — CV model inference
- `TorchXRayVision` — Pretrained chest X-ray model

**Infrastructure**
- `Docker` — Containerized backend
- `HuggingFace Spaces` — Backend deployment (free, 16GB RAM)
- `Vercel` — Frontend deployment (free)
- `HuggingFace Hub` — Model weights hosting

**Frontend**
- `React + Vite`
- `TailwindCSS v4`
- `React Router`
- `React Dropzone`
- `React Markdown`

---

## Multi-Step Intake Form

The frontend guides the user through a clinical intake flow before analysis:

**Step 1 — Patient Info + Scan Type**
Patient name + select scan type (Chest X-Ray / Skin Lesion / Retinal)

**Step 2 — Clinical Questions**
Condition-specific questions:
- Skin: body location, duration, changes, sun exposure history
- X-Ray: symptoms, duration, medical history
- Retinal: diabetes type, duration, HbA1c, last eye exam

**Step 3 — Upload Image**
Drag-and-drop or click to upload (JPG/PNG)

**Step 4 — Structured Report**
Sections: Scan Summary · Detailed Findings · Clinical Reasoning · Location Analysis · Risk Assessment · Recommendations · Referral · Disclaimer

---

## Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- Groq API key (free at console.groq.com)

### Backend

```bash
git clone https://github.com/MuhammadAbdulrehman-bit/MedPanel.git
cd MedPanel

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies (CPU PyTorch)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

# Set environment variables
cp .env.example .env
# Add your GROQ_API_KEY to .env

# Download model weights from HuggingFace
python download_weights.py

# Build RAG knowledge base
python -m backend.rag.ingest

# Run server
uvicorn backend.main:app --reload
```

### Frontend

```bash
cd frontend/mediscan-web
npm install
npm run dev
```

Open `http://localhost:5173`

---

## API Reference

### POST `/analyze`

Analyze a medical image through the full agent pipeline.

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | image | ✅ | JPG or PNG medical image |
| `patient_name` | string | ✅ | Patient full name |
| `scan_hint` | string | ✅ | `xray` / `skin` / `retinal` |
| `body_location` | string | ⬜ | Body location (critical for skin scans) |

**Response:**
```json
{
  "success": true,
  "patient_name": "John Doe",
  "scan_type": "skin_lesion",
  "top_finding": "Melanocytic Nevus",
  "risk_level": "Medium",
  "findings": [
    {"condition": "Melanocytic Nevus", "confidence": 0.983},
    {"condition": "Melanoma", "confidence": 0.016}
  ],
  "report": "## 📋 SCAN SUMMARY\n..."
}
```

### GET `/health`
Returns `{"status": "healthy"}`

---

## Project Structure

```
MedPanel/
├── backend/
│   ├── agents/
│   │   ├── vision_agent.py      # Agent 1 — CV routing + inference
│   │   ├── rag_agent.py         # Agent 2 — ChromaDB retrieval
│   │   └── report_agent.py      # Agent 3 — Groq reasoning
│   ├── models/
│   │   ├── xray_model.py        # TorchXRayVision DenseNet121
│   │   ├── skin_model.py        # EfficientNetB3 (trained)
│   │   └── retinal_model.py     # ResNet50 (trained)
│   ├── pipeline/
│   │   └── orchestrator.py      # LangGraph state machine
│   ├── rag/
│   │   ├── ingest.py            # ChromaDB knowledge base builder
│   │   └── knowledge/           # 20 disease knowledge files
│   ├── db/
│   │   └── database.py
│   └── main.py                  # FastAPI application
├── frontend/
│   └── mediscan-web/            # React + Vite frontend
├── download_weights.py          # Downloads weights from HuggingFace
├── Dockerfile
└── docker-compose.yml
```

---

## Disclaimer

MediScan AI is a research and portfolio project demonstrating multi-agent AI architecture applied to medical imaging. It is **not** a medical device and should **not** be used for clinical diagnosis. All outputs must be reviewed by qualified medical professionals.

---

## Author

**Muhammad Abdul Rehman**
BS Artificial Intelligence — Shifa Tameer-e-Millat University, Islamabad
[GitHub](https://github.com/MuhammadAbdulrehman-bit) · [LinkedIn](https://www.linkedin.com/in/muhammad-abdul-rehman-0bb2b02aa/)
