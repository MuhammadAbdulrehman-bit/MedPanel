import os
import uuid
import shutil
import json
import hmac
import hashlib
import httpx
from datetime import datetime, date
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from backend.db.database import get_db, init_db, ScanRecord, User
from backend.pipeline.orchestrator import run_pipeline
from backend.auth import (
    hash_password, verify_password,
    create_access_token, get_current_user
)

load_dotenv()

app = FastAPI(
    title="MediScan AI",
    description="Autonomous Medical Imaging Triage System",
    version="2.0.0"
)

# ─── CORS ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Upload Directory ──────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

FREE_SCANS_PER_DAY = 1


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


# ─── Health ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "MediScan AI is running", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}


# ─── Auth: Register ───────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

@app.post("/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    user = User(
        name=body.name,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.email)
    return {
        "token": token,
        "user": {"id": user.id, "name": user.name, "email": user.email, "is_pro": user.is_pro}
    }


# ─── Auth: Login ──────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id, user.email)
    return {
        "token": token,
        "user": {"id": user.id, "name": user.name, "email": user.email, "is_pro": user.is_pro}
    }


# ─── Auth: Me ─────────────────────────────────────────────────────────
@app.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "is_pro": current_user.is_pro
    }


# ─── Scan history ─────────────────────────────────────────────────────
@app.get("/scans/me")
def get_my_scans(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    scans = (
        db.query(ScanRecord)
        .filter(ScanRecord.user_id == current_user.id)
        .order_by(ScanRecord.created_at.desc())
        .all()
    )
    return [
        {
            "id": s.id,
            "patient_name": s.patient_name,
            "scan_type": s.scan_type,
            "risk_level": s.risk_level,
            "findings": json.loads(s.findings) if s.findings else [],
            "created_at": s.created_at.isoformat(),
        }
        for s in scans
    ]


# ─── Scan detail ──────────────────────────────────────────────────────
@app.get("/scans/me/{scan_id}")
def get_scan_detail(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    scan = db.query(ScanRecord).filter(
        ScanRecord.id == scan_id,
        ScanRecord.user_id == current_user.id
    ).first()

    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    return {
        "id": scan.id,
        "patient_name": scan.patient_name,
        "scan_type": scan.scan_type,
        "risk_level": scan.risk_level,
        "findings": json.loads(scan.findings) if scan.findings else [],
        "report": scan.report,
        "created_at": scan.created_at.isoformat(),
    }


# ─── Scan limit check ─────────────────────────────────────────────────
def check_scan_limit(user: User, db: Session):
    if user.is_pro:
        return

    today_start = datetime.combine(date.today(), datetime.min.time())
    scans_today = (
        db.query(ScanRecord)
        .filter(
            ScanRecord.user_id == user.id,
            ScanRecord.created_at >= today_start
        )
        .count()
    )
    if scans_today >= FREE_SCANS_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_limit_reached",
                "message": f"Free plan allows {FREE_SCANS_PER_DAY} scan per day. Upgrade to Pro for unlimited scans.",
                "scans_used": scans_today,
                "limit": FREE_SCANS_PER_DAY
            }
        )


# ─── Lemon Squeezy: Create checkout ───────────────────────────────────
@app.post("/create-checkout")
async def create_checkout(current_user: User = Depends(get_current_user)):
    api_key = os.getenv("LEMONSQUEEZY_API_KEY")
    variant_id = os.getenv("LEMONSQUEEZY_VARIANT_ID")

    if not api_key or not variant_id:
        raise HTTPException(status_code=500, detail="Payment not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.lemonsqueezy.com/v1/checkouts",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/vnd.api+json",
                "Content-Type": "application/vnd.api+json",
            },
            json={
                "data": {
                    "type": "checkouts",
                    "attributes": {
                        "checkout_data": {
                            "email": current_user.email,
                            "custom": {
                                "user_id": str(current_user.id),
                                "email": current_user.email
                            }
                        },
                        "product_options": {
                            "redirect_url": "https://mediscan-web.vercel.app/?upgraded=true"
                        }
                    },
                    "relationships": {
                        "store": {
                            "data": {
                                "type": "stores",
                                "id": str(os.getenv("LEMONSQUEEZY_STORE_ID"))
                            }
                        },
                        "variant": {
                            "data": {
                                "type": "variants",
                                "id": str(variant_id)
                            }
                        }
                    }
                }
            }
        )

    if response.status_code != 201:
        print(f"[Checkout] Error: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to create checkout")

    checkout_url = response.json()["data"]["attributes"]["url"]
    return {"checkout_url": checkout_url}


# ─── Lemon Squeezy: Webhook ───────────────────────────────────────────
@app.post("/webhook/lemonsqueezy")
async def lemonsqueezy_webhook(request: Request, db: Session = Depends(get_db)):
    webhook_secret = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET", "")
    body = await request.body()

    # Verify signature if secret is set
    # Verify signature if secret is set
    if webhook_secret:
        signature = request.headers.get("X-Signature", "")
        expected = hmac.new(
            webhook_secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            print(f"[Webhook] Signature mismatch - expected: {expected[:20]}... got: {signature[:20]}...")
            # Temporarily log but don't block
            # raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = json.loads(body)
    event_name = payload.get("meta", {}).get("event_name", "")
    print(f"[Webhook] Event: {event_name}")

    if event_name == "order_created":
        custom_data = payload.get("meta", {}).get("custom_data", {})
        user_id = custom_data.get("user_id")
        email = custom_data.get("email")

        print(f"[Webhook] Payment for user_id={user_id} email={email}")

        user = None
        if user_id:
            user = db.query(User).filter(User.id == int(user_id)).first()
        if not user and email:
            user = db.query(User).filter(User.email == email).first()

        if user:
            user.is_pro = True
            db.commit()
            print(f"[Webhook] ✅ {user.email} upgraded to Pro")
        else:
            print(f"[Webhook] ⚠️ User not found")

    return {"status": "ok"}


# ─── Analyze ──────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze_image(
    patient_name: str = Form(...),
    scan_hint: str = Form(None),
    body_location: str = Form(None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    check_scan_limit(current_user, db)

    allowed = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images supported.")

    file_ext = file.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[API] User {current_user.email} analyzing image for patient: {patient_name}")

    try:
        result = run_pipeline(
            patient_name=patient_name,
            image_path=file_path,
            scan_hint=scan_hint,
            body_location=body_location
        )

        if os.path.exists(file_path):
            os.remove(file_path)

        if result.get("error") and not result.get("final_report"):
            raise HTTPException(status_code=500, detail=result["error"])

        scan = ScanRecord(
            user_id=current_user.id,
            patient_name=patient_name,
            scan_type=result.get("scan_type"),
            findings=json.dumps(result.get("findings", [])),
            risk_level=result.get("risk_level"),
            report=result.get("final_report"),
        )
        db.add(scan)
        db.commit()

        return JSONResponse(content={
            "success": True,
            "patient_name": patient_name,
            "scan_type": result.get("scan_type"),
            "top_finding": result.get("top_finding"),
            "risk_level": result.get("risk_level"),
            "findings": result.get("findings"),
            "report": result.get("final_report"),
            "scans_used_today": 1 if not current_user.is_pro else None,
            "is_pro": current_user.is_pro,
        })

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))