from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── NEW: User table ──────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_pro = Column(Boolean, default=False)           # False = free tier (1 scan/day)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─── UPDATED: ScanRecord now linked to a user ────────────────────────
class ScanRecord(Base):
    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)          # nullable for backward compat
    patient_name = Column(String, nullable=False)
    scan_type = Column(String, nullable=False)        # xray / skin / retinal
    image_url = Column(String, nullable=True)         # Cloudinary URL
    findings = Column(Text, nullable=True)            # CV model output
    risk_level = Column(String, nullable=True)        # Low / Medium / High
    report = Column(Text, nullable=True)              # Final generated report
    created_at = Column(DateTime, default=datetime.utcnow)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    topic = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
