import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from backend.db.database import get_db, init_db, ScanRecord
from backend.pipeline.orchestrator import run_pipeline

load_dotenv()

app = FastAPI(
    title="MediScan AI",
    description="Autonomous Medical Imaging Triage System",
    version="1.0.0"
)

# ─── CORS ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Upload Directory ─────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── Startup ──────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    print("[MediScan] Starting up...")
    try:
        init_db()
        print("[MediScan] Database initialized.")
    except Exception as e:
        print(f"[MediScan] DB not available yet: {e}")
    print("[MediScan] Ready.")


# ─── Health Check ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "MediScan AI is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ─── Main Analyze Endpoint ────────────────────────────────────────────
@app.post("/analyze")
async def analyze_image(
    patient_name: str = Form(...),
    scan_hint: str = Form(None),
    file: UploadFile = File(...)
):
    """
    Main endpoint. Accepts a medical image and runs the full
    3-agent pipeline: Vision → RAG → Report.
    """
    # Validate file type
    allowed = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported."
        )

    # Save uploaded file temporarily
    file_ext = file.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[API] Received image: {file.filename} for patient: {patient_name}")

    try:
        # Run the full pipeline
        result = run_pipeline(
            patient_name=patient_name,
            image_path=file_path,
            scan_hint=scan_hint
        )

        # Clean up temp file
        os.remove(file_path)

        if result.get("error") and not result.get("final_report"):
            raise HTTPException(status_code=500, detail=result["error"])

        return JSONResponse(content={
            "success": True,
            "patient_name": patient_name,
            "scan_type": result.get("scan_type"),
            "top_finding": result.get("top_finding"),
            "risk_level": result.get("risk_level"),
            "findings": result.get("findings"),
            "report": result.get("final_report"),
            "error": result.get("error")
        })

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Get All Scans (from DB — enabled after Docker setup) ─────────────
@app.get("/scans")
def get_scans():
    """Returns all past scan records. Requires DB to be running."""
    return {"message": "Database not connected yet. Available after Docker setup."}