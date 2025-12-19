import os
import json
import csv
from io import StringIO
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Any

# FastAPI
from fastapi import FastAPI, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Pydantic
from pydantic import BaseModel, EmailStr

# SQLAlchemy (async)
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, ForeignKey, Text,
    UniqueConstraint, select, or_
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select as async_select


# Password hashing
from passlib.context import CryptContext

# JWT
from jose import jwt, JWTError

# Google Login
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# AWS S3
import aioboto3
import boto3
from botocore.exceptions import ClientError

# Load env
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging

logger = logging.getLogger("app")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# concurrency limit for simultaneous S3 calls
S3_CONCURRENCY = 20

SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = int(os.getenv("ACCESS_TOKEN_EXPIRE_SECONDS", 3600))

DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

TASKS_PER_USER = int(os.getenv("TASKS_PER_USER", "50"))
# SESSION_MINUTES = int(os.getenv("SESSION_MINUTES", "60"))
PASSING_SCORE = int(os.getenv("PASSING_SCORE", "50"))

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@demo.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# AWS ENV
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET", "agriasrteluguv1")

# default testing prefix used in listing and json resolution
DEFAULT_TESTING_PREFIX = "Testingsakeeth/"
print(DATABASE_URL)
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be set.")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise RuntimeError("AWS credentials missing.")

# Patch DB URL for asyncpg if needed
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql+psycopg2://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)

app = FastAPI(title="Main System with AWS S3 & Metadata Fix (Final)")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{path:path}")
async def preflight_handler(path: str):
    return JSONResponse(status_code=200)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0
    }
)


AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    full_name = Column(String, nullable=False)
    college_id = Column(String, nullable=False)
    college_name = Column(String, nullable=False)
    mobile_number = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)

    gender = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    dialect = Column(String, nullable=True)
    mode = Column(String, nullable=True)

    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session_start = Column(DateTime(timezone=True), nullable=True)
    session_end = Column(DateTime(timezone=True), nullable=True)

    assignments = relationship("FileAssignment", back_populates="user", cascade="all, delete-orphan")

class FileAssignment(Base):
    __tablename__ = "file_assignments"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    base = Column(String, nullable=False)

    audio_key = Column(String, nullable=False)
    text_key = Column(String, nullable=False)
    json_key = Column(String, nullable=True)

    assigned_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    action = Column(String, nullable=True)
    edited_text = Column(Text, nullable=True)
    score = Column(Integer, default=0)
    completed = Column(Boolean, default=False)

    user = relationship("User", back_populates="assignments")

    __table_args__ = (UniqueConstraint("user_id", "base", name="uq_user_task"),)

# =========================
# AUTH HELPERS
# =========================

def create_token(data: dict, expires_seconds: int = ACCESS_TOKEN_EXPIRE_SECONDS):
    payload = data.copy()
    expire = datetime.utcnow() + timedelta(seconds=expires_seconds)
    payload["exp"] = expire
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

def require_admin(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    payload = decode_token(token)
    if not payload or not payload.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return payload

# =========================
# AWS S3 HELPERS
# =========================

def get_aioboto_session():
    return aioboto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

def get_boto_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

async def s3_object_exists(key: str) -> bool:
    if not key:
        return False
    session = get_aioboto_session()
    async with session.client("s3") as s3:
        try:
            await s3.head_object(Bucket=S3_BUCKET, Key=key)
            return True
        except ClientError:
            return False
        except Exception:
            logger.exception("Error checking existence of %s", key)
            return False

async def read_json_from_s3(key: str) -> Any:
    if not key:
        return None
    session = get_aioboto_session()
    async with session.client("s3") as s3:
        try:
            resp = await s3.get_object(Bucket=S3_BUCKET, Key=key)
            body = await resp["Body"].read()
            return json.loads(body.decode("utf-8"))
        except Exception:
            logger.exception("Failed to read JSON %s", key)
            return None

async def write_json_to_s3(key: str, data: Any):
    """
    Writes JSON to S3. Will create file if needed (used for incorrect/ files only).
    Avoid writing empty dict/list unless it is meaningful.
    """
    if not key or data is None:
        return
    # do not write empty list/dict
    if (isinstance(data, dict) and len(data) == 0) or (isinstance(data, list) and len(data) == 0):
        logger.info("Refusing to write empty JSON for key %s", key)
        return
    session = get_aioboto_session()
    async with session.client("s3") as s3:
        await s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

def _audio_filename_from_key(key: str) -> str:
    """
    Return the last path component (basename).
    """
    if not key:
        return ""
    return key.split("/")[-1]

def _match_audio_filename_in_entry(entry: Any, audio_filename: str) -> bool:
    """
    Strict match:
    - If entry has 'audio_filepath' (or 'audio'/'audio_path'), check if it endswith audio_filename.
    - Use basename equality or endswith match.
    """
    if not isinstance(entry, dict) or not audio_filename:
        return False

    for field in ("audio_filepath", "audio", "audio_path", "filename"):
        val = entry.get(field)
        if isinstance(val, str) and val:
            # exact basename equality
            if val.split("/")[-1] == audio_filename:
                return True
            # endswith (allow absolute paths where tail equals)
            if val.endswith(audio_filename):
                return True
    return False

async def resolve_main_json_for_folder(audio_key_hint: Optional[str]) -> Optional[str]:
    """
    Resolve main JSON strictly as Testingsakeeth/<Folder>.json derived from audio_key_hint.
    Does NOT attempt other keys or create files. Returns None if not found.
    """
    if not audio_key_hint:
        return None
    parts = audio_key_hint.split("/")
    # expected structure: Testingsakeeth/Folder00009/<file>.wav  (or maybe other top-level but we enforce second part)
    if len(parts) < 2:
        return None
    folder = parts[1]
    candidate = f"{DEFAULT_TESTING_PREFIX}{folder}.json"
    if await s3_object_exists(candidate):
        return candidate
    logger.info("Main JSON not found (create not allowed). Expected %s", candidate)
    return None

async def remove_entry_from_main_json(main_json_key: str, audio_key: str) -> Optional[dict]:
    """
    Remove the matching entry (by filename) from the main JSON (list) and return removed dict.
    Strict behavior: main_json_key must exist and be a list. Matching by basename/endswith only.
    """
    if not main_json_key:
        logger.error("remove_entry_from_main_json called without main_json_key")
        return None

    data = await read_json_from_s3(main_json_key)
    if data is None:
        logger.error("Main JSON %s could not be read (expected to exist)", main_json_key)
        return None

    audio_filename = _audio_filename_from_key(audio_key)
    removed_entry = None

    if isinstance(data, list):
        new_list = []
        for entry in data:
            if isinstance(entry, dict) and _match_audio_filename_in_entry(entry, audio_filename):
                removed_entry = entry
                # skip adding to new_list (i.e., remove)
                continue
            new_list.append(entry)
        if removed_entry:
            # write back updated main json (only if something removed)
            await write_json_to_s3(main_json_key, new_list)
            return removed_entry
        # not found
        logger.info("No matching entry found in %s for audio %s", main_json_key, audio_filename)
        return None
    else:
        logger.error("Main JSON %s is not a list; unsupported structure under 'must-exist' rule", main_json_key)
        return None

async def append_entry_to_incorrect(main_json_key: str, audio_key: str, entry: dict):
    """
    Append removed entry into incorrect/Testingsakeeth/<Folder>.json
    - main_json_key used to derive folder name for correct incorrect path
    - If incorrect file does not exist, create it as a list with single entry
    """
    if not entry:
        logger.info("No entry provided to append to incorrect JSON for %s", audio_key)
        return

    # derive folder json name strictly from main_json_key
    if not main_json_key:
        logger.info("No main json key provided to derive incorrect path; skipping append")
        return

    # main_json_key expected like "Testingsakeeth/Folder00009.json"
    if main_json_key.startswith(DEFAULT_TESTING_PREFIX):
        rel = main_json_key[len(DEFAULT_TESTING_PREFIX):]
        folder_json_name = rel  # e.g., Folder00009.json
    else:
        folder_json_name = os.path.basename(main_json_key)

    incorrect_json_key = f"incorrect/{DEFAULT_TESTING_PREFIX}{folder_json_name}"

    existing = await read_json_from_s3(incorrect_json_key)

    payload = entry.copy()
    if "audio_filepath" not in payload:
        payload["audio_filepath"] = audio_key

    if existing is None:
        await write_json_to_s3(incorrect_json_key, [payload])
        return

    if isinstance(existing, list):
        existing.append(payload)
        await write_json_to_s3(incorrect_json_key, existing)
        return

    if isinstance(existing, dict):
        new_list = [existing, payload]
        await write_json_to_s3(incorrect_json_key, new_list)
        return

    await write_json_to_s3(incorrect_json_key, [payload])

# =========================
# METADATA UPDATER (strict must-exist behavior)
# =========================

async def update_metadata_in_existing_main_json(
    audio_key_hint: Optional[str],
    gender: Optional[str] = None,
    age: Optional[Any] = None,
    dialect: Optional[str] = None,
    mode: Optional[str] = None,
    edited_text: Optional[str] = None,
):
    """
    Strict update:
      - Resolve JSON strictly via resolve_main_json_for_folder(audio_key_hint)
      - If not found -> skip
      - Match entries only when entry.audio_filepath endswith basename(audio_key_hint)
      - Update only the provided fields
    """
    if not audio_key_hint:
        logger.info("No audio hint provided for update; skipping.")
        return

    resolved = await resolve_main_json_for_folder(audio_key_hint)
    if not resolved:
        # strict: do not create main JSON
        logger.info("Main JSON not found (create not allowed). Skipping metadata update for %s", audio_key_hint)
        return

    data = await read_json_from_s3(resolved)
    if data is None:
        logger.warning("Failed to read main JSON %s", resolved)
        return

    if not isinstance(data, list):
        logger.warning("Main JSON %s is not list; skipping update", resolved)
        return

    audio_filename = _audio_filename_from_key(audio_key_hint)
    updated = False

    for entry in data:
        if not isinstance(entry, dict):
            continue
        if _match_audio_filename_in_entry(entry, audio_filename):
            # Only update specified fields
            if gender is not None:
                entry["Gender"] = gender
            if age is not None and age != "":
                try:
                    entry["age"] = int(age)
                except Exception:
                    entry["age"] = age
            if dialect is not None:
                entry["Dialect"] = dialect
            if mode is not None:
                entry["Mode"] = mode
            if edited_text is not None:
                entry["edited_text"] = edited_text
            updated = True
            break

    if updated:
        await write_json_to_s3(resolved, data)
    else:
        logger.info("No matching entry found in %s for audio %s — skipping update", resolved, audio_key_hint)

# =========================
# S3 listing / read / write / presign / move helpers
# =========================

async def list_pairs_s3(testing_prefix: Optional[str] = DEFAULT_TESTING_PREFIX):
    session = get_aioboto_session()
    audio_keys = set()
    text_keys = set()
    async with session.client("s3") as s3:
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=testing_prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                base, ext = os.path.splitext(key)
                if ext.lower() in AUDIO_EXTS:
                    audio_keys.add(key)
                elif ext.lower() == ".txt":
                    text_keys.add(key)

    pairs = []
    txt_set = set(text_keys)
    for audio_key in sorted(audio_keys):
        base_key = audio_key.rsplit(".", 1)[0]
        text_key = base_key + ".txt"
        if text_key not in txt_set:
            continue
        parts = base_key.split("/")
        json_key = None
        if len(parts) >= 2:
            folder = parts[1]
            json_key = f"{DEFAULT_TESTING_PREFIX}{folder}.json"
        pairs.append((base_key, audio_key, text_key, json_key))
    return pairs

async def read_text_from_s3(key: str) -> str:
    session = get_aioboto_session()
    async with session.client("s3") as s3:
        try:
            resp = await s3.get_object(Bucket=S3_BUCKET, Key=key)
            body = await resp["Body"].read()
            return body.decode("utf-8")
        except Exception:
            return ""

def get_presigned_url(key: str, expires: int = 3600):
    client = get_boto_client()
    try:
        url = client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires
        )
        return url
    except Exception:
        return None

async def write_text_to_s3(key: str, text: str):
    session = get_aioboto_session()
    async with session.client("s3") as s3:
        await s3.put_object(Bucket=S3_BUCKET, Key=key, Body=text.encode("utf-8"), ContentType="text/plain")

async def move_files_to_incorrect(keys: List[str]):
    """
    Move given keys under incorrect/<original_key>.
    If moving fails for a given key, log and continue.
    """
    session = get_aioboto_session()
    async with session.client("s3") as s3:
        for key in keys:
            try:
                dest_key = f"incorrect/{key}"
                copy_source = {"Bucket": S3_BUCKET, "Key": key}
                await s3.copy_object(Bucket=S3_BUCKET, CopySource=copy_source, Key=dest_key)
                await s3.delete_object(Bucket=S3_BUCKET, Key=key)
            except Exception:
                logger.exception("Failed to move key %s to incorrect/ prefix", key)

async def move_files_to_incorrect_and_update_metadata(keys: List[str], audio_key: str):
    """
    Moves audio and text to incorrect/ prefix.
    Attempts to remove metadata from main JSON and append to incorrect JSON only if main JSON exists.
    Strict behavior: if main JSON not found, skip metadata changes.
    """
    # Move files regardless
    await move_files_to_incorrect(keys)

    # Try metadata removal only if main json exists
    main_json = await resolve_main_json_for_folder(audio_key)
    if not main_json:
        logger.info("Main JSON not found while moving files to incorrect; skipping metadata move for %s", audio_key)
        return

    removed_entry = await remove_entry_from_main_json(main_json, audio_key)
    if removed_entry:
        await append_entry_to_incorrect(main_json, audio_key, removed_entry)
    else:
        logger.info("No metadata entry removed for %s — no append to incorrect JSON", audio_key)

# =========================
# Auto-submit
# =========================

async def auto_submit(user_id: int, db: AsyncSession):
    logger.info("Auto-submitting session for user %s", user_id)

    res = await db.execute(
        async_select(FileAssignment).filter(
            FileAssignment.user_id == user_id,
            FileAssignment.completed == False
        )
    )
    pending_tasks = res.scalars().all()

    for task in pending_tasks:
        task.completed = True
        task.action = "auto_submitted"
        task.score = 0
        task.completed_at = datetime.now(timezone.utc)

    res_user = await db.execute(async_select(User).filter(User.id == user_id))
    user = res_user.scalars().first()
    if user:
        user.session_end = datetime.now(timezone.utc)

    await db.commit()
    logger.info("Auto-submit complete for user %s", user_id)
    return {"done": True, "message": "Session auto-submitted successfully"}

# =========================
# Schemas
# =========================
async def get_unassigned_tasks(db: AsyncSession):
    """
    Returns list of unassigned tasks from S3 that are not yet present
    in file_assignments table.
    """

    # 1. Get all S3 task pairs
    s3_pairs = await list_pairs_s3()

    # 2. Get already assigned bases from DB
    res = await db.execute(select(FileAssignment.base))
    assigned_bases = {row[0] for row in res.all()}

    # 3. Filter unassigned tasks
    unassigned = []
    for base, audio_key, text_key, json_key in s3_pairs:
        if base not in assigned_bases:
            unassigned.append({
                "base": base,
                "audio_key": audio_key,
                "text_key": text_key,
                "json_key": json_key
            })

    return unassigned

class RegisterIn(BaseModel):
    full_name: str
    college_id: str
    college_name: str
    mobile_number: str
    email: EmailStr

class TokenRequest(BaseModel):
    token: str

class AssignTasksIn(BaseModel):
    user_id: int
    limit: int|None=None

class CorrectTaskIn(BaseModel):
    user_id: int
    base: str
    gender: Optional[str] = None
    age: Optional[str] = None
    dialect: Optional[str] = None
    mode: Optional[str] = None

class EditTaskIn(BaseModel):
    user_id: int
    base: str
    edited_text: str
    gender: Optional[str] = None
    age: Optional[str] = None
    dialect: Optional[str] = None
    mode: Optional[str] = None

class IncorrectTaskIn(BaseModel):
    user_id: int
    base: str
    gender: Optional[str] = None
    age: Optional[str] = None
    dialect: Optional[str] = None
    mode: Optional[str] = None

class SaveProgressIn(BaseModel):
    user_id: int
    score: int
    current: int
    force_submit: bool = False

class AdminLoginIn(BaseModel):
    email: str
    password: str
# =========================
# Endpoints
# =========================

@app.post("/register")
async def register(data: RegisterIn, db: AsyncSession = Depends(get_db)):
    if not data.email.endswith("@gmail.com"):
        raise HTTPException(status_code=400, detail="Only Gmail accounts are allowed")

    res = await db.execute(select(User).filter(User.email == data.email))
    if res.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(**data.dict())
    db.add(user)
    await db.commit()
    return {"message": "registered", "email": data.email}

@app.post("/google-login")
async def google_login(request: TokenRequest, db: AsyncSession = Depends(get_db)):
    token = request.token
    try:
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo["email"]

        res = await db.execute(select(User).filter(User.email == email))
        user = res.scalars().first()
        if not user:
            raise HTTPException(status_code=403, detail="Email not registered in the system")

        res2 = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == user.id))
        tasks = res2.scalars().all()
        if len(tasks) == 0:
            pairs = await list_pairs_s3()
            used_rows = await db.execute(select(FileAssignment.base))
            used_bases = {r[0] for r in used_rows.all()}
            available_pairs = [p for p in pairs if p[0] not in used_bases]
            new_pairs = available_pairs[:TASKS_PER_USER]
            if not new_pairs:
                raise HTTPException(status_code=400, detail="No tasks available")
            for base, audio_key, text_key, json_key in new_pairs:
                db.add(FileAssignment(user_id=user.id, base=base, audio_key=audio_key, text_key=text_key, json_key=json_key))
            await db.commit()

        jwt_token = create_token({"id": user.id, "email": user.email, "is_admin": user.is_admin})
        return {
            "access_token": jwt_token,
            "token_type": "bearer",
            "user_id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "college_name": user.college_name,
            "mobile_number": user.mobile_number
        }
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google token")

@app.get("/user/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id,
        "full_name": user.full_name,
        "college_id": user.college_id,
        "college_name": user.college_name,
        "mobile_number": user.mobile_number,
        "email": user.email,
        "gender": user.gender,
        "age": user.age,
        "dialect": user.dialect,
        "mode": user.mode
    }

@app.get("/dashboard/{user_id}")
async def get_dashboard(user_id: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == user_id))
    tasks = res.scalars().all()
    total = len(tasks)
    completed = sum(1 for t in tasks if t.completed)
    correct = sum(1 for t in tasks if t.action == "correct")
    incorrect = sum(1 for t in tasks if t.action == "incorrect")
    edited = sum(1 for t in tasks if t.action == "edit")
    score = sum(t.score for t in tasks)
    status = "Selected" if score >= PASSING_SCORE else "Unselected"
    return {
        "total": total,
        "completed": completed,
        "correct": correct,
        "incorrect": incorrect,
        "edited": edited,
        "score": score,
        "status": status
    }

@app.get("/start_session/{user_id}")
async def start_session(user_id: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).filter(User.id == user_id))
    user = res.scalars().first()
    if not user:
        return JSONResponse({"locked": True, "reason": "User not found"}, status_code=404)

    now = datetime.now(timezone.utc)

    if not user.session_start:
        user.session_start = now
        # user.session_end = now + timedelta(minutes=SESSION_MINUTES)
        user.session_end = None
        await db.commit()

    # if user.session_end and now > user.session_end:
    #     await auto_submit(user_id, db)
    #     return JSONResponse({"locked": True, "reason": "Session expired — test auto-submitted"}, status_code=200)

    res_tasks = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == user_id))
    all_tasks = res_tasks.scalars().all()
    total = len(all_tasks)
    completed = sum(1 for t in all_tasks if t.completed)

    if completed >= TASKS_PER_USER:
        await auto_submit(user_id, db)
        return JSONResponse({"locked": True, "reason": f"All {TASKS_PER_USER} tasks completed — test ended"}, status_code=200)

    res_task = await db.execute(
        select(FileAssignment)
        .filter(FileAssignment.user_id == user_id, FileAssignment.completed == False)
        .order_by(FileAssignment.id.asc())
        .limit(1)
    )
    task = res_task.scalars().first()
    if not task:
        await auto_submit(user_id, db)
        return JSONResponse({"locked": True, "reason": "No remaining tasks — test ended"}, status_code=200)

    text_content = await read_text_from_s3(task.text_key)
    audio_url = get_presigned_url(task.audio_key)

    remaining = (user.session_end - now).total_seconds() if user.session_end else 0
    return JSONResponse({
        "locked": False,
        "base": task.base,
        "audio_url": audio_url,
        "text_blob": text_content,
        "total_tasks": total,
        "completed_tasks": completed,
        "time_remaining": int(remaining),
        "session_end": user.session_end.isoformat() if user.session_end else None,
        "passing_score": PASSING_SCORE,
    }, status_code=200)

# helper: next task
async def _next_task(user_id: int, db: AsyncSession):
    res = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == user_id, FileAssignment.completed == False))
    next_task = res.scalars().first()
    if not next_task:
        res_user = await db.execute(select(User).filter(User.id == user_id))
        user = res_user.scalars().first()
        if user:
            user.session_end = datetime.now(timezone.utc)
        await db.commit()
        return {"done": True}

    text = await read_text_from_s3(next_task.text_key)
    audio_url = get_presigned_url(next_task.audio_key)

    return {
        "base": next_task.base,
        "audio_url": audio_url,
        "text": text
    }

# -----------------------
# Mark correct
# -----------------------
@app.post("/correct")
async def mark_correct(
    data: CorrectTaskIn,
    db: AsyncSession = Depends(get_db)
):
    user_id = data.user_id
    base = data.base
    gender = data.gender
    age = data.age
    dialect = data.dialect
    mode = data.mode
    res = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == user_id, FileAssignment.base == base))
    task = res.scalars().first()
    if not task:
        raise HTTPException(404, "Task not found")
    if task.completed:
        raise HTTPException(400, "Already completed")

    parsed_age = None
    if age is not None and age != "":
        try:
            parsed_age = int(age)
        except Exception:
            parsed_age = age

    # Strict update only in existing main json — create not allowed
    await update_metadata_in_existing_main_json(
        audio_key_hint=task.audio_key,
        gender=gender,
        age=parsed_age,
        dialect=dialect,
        mode=mode,
        edited_text=None
    )

    task.action = "correct"
    task.score = 1
    task.completed = True
    task.completed_at = datetime.now(timezone.utc)
    await db.commit()

    return await _next_task(user_id, db)

# -----------------------
# Mark edit
# -----------------------
@app.post("/edit")
async def mark_edit(
    data: EditTaskIn,
    db: AsyncSession = Depends(get_db)
):
    user_id = data.user_id
    base = data.base
    edited_text = data.edited_text
    gender = data.gender
    age = data.age
    dialect = data.dialect
    mode = data.mode
    res = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == user_id, FileAssignment.base == base))
    task = res.scalars().first()
    if not task:
        raise HTTPException(404, "Task not found")
    if task.completed:
        raise HTTPException(400, "Task already completed")

    # write updated text to S3
    await write_text_to_s3(task.text_key, edited_text)

    parsed_age = None
    if age is not None and age != "":
        try:
            parsed_age = int(age)
        except Exception:
            parsed_age = age

    # Update metadata in existing main JSON only
    await update_metadata_in_existing_main_json(
        audio_key_hint=task.audio_key,
        gender=gender,
        age=parsed_age,
        dialect=dialect,
        mode=mode,
        edited_text=edited_text
    )

    task.action = "edit"
    task.score = 1
    task.edited_text = edited_text
    task.completed = True
    task.completed_at = datetime.now(timezone.utc)
    await db.commit()

    return await _next_task(user_id, db)

# -----------------------
# Mark incorrect
# -----------------------
@app.post("/incorrect")
async def mark_incorrect(
    data: IncorrectTaskIn,
    db: AsyncSession = Depends(get_db)
):
    user_id = data.user_id
    base = data.base
    gender = data.gender
    age = data.age
    dialect = data.dialect
    mode = data.mode
    res = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == user_id, FileAssignment.base == base))
    task = res.scalars().first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.completed:
        raise HTTPException(status_code=400, detail="Task already completed")

    parsed_age = None
    if age is not None and age != "":
        try:
            parsed_age = int(age)
        except Exception:
            parsed_age = age

    # Update fields in main json BEFORE moving (if provided). Strict: only existing main json updated.
    await update_metadata_in_existing_main_json(
        audio_key_hint=task.audio_key,
        gender=gender,
        age=parsed_age,
        dialect=dialect,
        mode=mode,
        edited_text=None
    )

    keys_to_move = [task.audio_key, task.text_key]
    await move_files_to_incorrect_and_update_metadata(keys_to_move, task.audio_key)

    task.action = "incorrect"
    task.score = 1
    task.completed = True
    task.completed_at = datetime.now(timezone.utc)
    await db.commit()

    return await _next_task(user_id, db)

# -----------------------
# Save progress
# -----------------------
@app.post("/save_progress")
async def save_progress(
    data: SaveProgressIn,
    db: AsyncSession = Depends(get_db)
):
    user_id = data.user_id
    score = data.score
    current = data.current
    force_submit = data.force_submit
    res = await db.execute(select(User).filter(User.id == user_id))
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    res_tasks = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == user_id))
    tasks = res_tasks.scalars().all()
    total = len(tasks)
    completed = sum(1 for t in tasks if t.completed)

    if completed >= TASKS_PER_USER or current >= TASKS_PER_USER:
        await auto_submit(user_id, db)
        return {"done": True, "message": f"All {TASKS_PER_USER} tasks completed — test auto-submitted."}

    if force_submit:
        await auto_submit(user_id, db)
        return {"done": True, "message": "User ended test — auto-submitted."}

    await db.commit()
    return {"message": "Progress saved successfully"}

# -----------------------
# Admin endpoints
# -----------------------
@app.post("/admin/login")
async def admin_login(data: AdminLoginIn):
    email = data.email
    password = data.password
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        token = create_token({"id": 0, "email": email, "is_admin": True})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid admin credentials")

@app.get("/admin/summary")
async def admin_summary(
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    res = await db.execute(select(User))
    users = res.scalars().all()

    total_users = len(users)
    selected_users = 0
    rejected_users = 0
    total_tasks = 0
    completed_tasks = 0

    registered = []

    for u in users:
        res_tasks = await db.execute(
            select(FileAssignment).filter(FileAssignment.user_id == u.id)
        )
        tasks = res_tasks.scalars().all()

        assigned = len(tasks)
        completed = sum(1 for t in tasks if t.completed)
        correct = sum(1 for t in tasks if t.action == "correct")
        incorrect = sum(1 for t in tasks if t.action == "incorrect")
        edited = sum(1 for t in tasks if t.action == "edit")
        score = sum(t.score for t in tasks)

        total_tasks += assigned
        completed_tasks += completed

        status = (
            "Selected"
            if assigned > 0 and score == assigned
            else "Rejected"
        )

        if status == "Selected":
            selected_users += 1
        else:
            rejected_users += 1

        registered.append({
            "full_name": u.full_name,
            "email": u.email,
            "mobile_number": u.mobile_number,
            "college_name": u.college_name,

            "tasks_assigned": assigned,
            "tasks_completed": completed,
            "tasks_correct": correct,
            "tasks_incorrect": incorrect,
            "tasks_edited": edited,

            "score": score,
            "status": status
        })

    throughput = (
        f"{completed_tasks}/{total_tasks}"
        if total_tasks > 0
        else "0/0"
    )

    return {
        "stats": {
            "registered_users": total_users,
            "selected_users": selected_users,
            "rejected_users": rejected_users,
            "tasks_assigned": total_tasks,
            "throughput": throughput
        },
        "registered_users": registered}

@app.get("/admin/users/registered")
async def get_registered_users(
    page: int = 1,
    search: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    query = select(User)

    if search:
        query = query.filter(
            or_(
                User.full_name.ilike(f"%{search}%"),
                User.email.ilike(f"%{search}%")
            )
        )

    if from_date and to_date:
        try:
            start_dt = datetime.strptime(from_date, "%Y-%m-%d")
            end_dt = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1, seconds=-1)
            query = query.filter(User.created_at >= start_dt, User.created_at <= end_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")
    elif from_date and not to_date:
        try:
            start_dt = datetime.strptime(from_date, "%Y-%m-%d")
            end_dt = start_dt + timedelta(days=1, seconds=-1)
            query = query.filter(User.created_at >= start_dt, User.created_at <= end_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")

    res_all = await db.execute(query.order_by(User.id.desc()))
    all_users = res_all.scalars().all()

    result = []
    for u in all_users:
        res_tasks = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == u.id))
        tasks = res_tasks.scalars().all()
        score = sum(t.score for t in tasks)
        result.append({
            "id": u.id,
            "name": u.full_name,
            "college_id": u.college_id,
            "college_name": u.college_name,
            "mobile_number": u.mobile_number,
            "email": u.email,
            "score": score if tasks else "---",
        })

    per_page = 14
    total_pages = (len(result) + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    paginated = result[start:end]

    return {
        "users": paginated,
        "total_pages": total_pages,
        "total_count": len(result)
    }

@app.get("/admin/users/selected-rejected")
async def get_selected_rejected_users(
    page: int = 1,
    status: Optional[str] = None,
    search: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    query = select(User)

    if search:
        query = query.filter(
            or_(
                User.full_name.ilike(f"%{search}%"),
                User.email.ilike(f"%{search}%")
            )
        )

    if from_date and to_date:
        try:
            start_dt = datetime.strptime(from_date, "%Y-%m-%d")
            end_dt = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1, seconds=-1)
            query = query.filter(User.created_at >= start_dt, User.created_at <= end_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")
    elif from_date and not to_date:
        try:
            start_dt = datetime.strptime(from_date, "%Y-%m-%d")
            end_dt = start_dt + timedelta(days=1, seconds=-1)
            query = query.filter(User.created_at >= start_dt, User.created_at <= end_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")

    res_all = await db.execute(query)
    all_users = res_all.scalars().all()

    result = []
    for u in all_users:
        res_tasks = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == u.id))
        tasks = res_tasks.scalars().all()
        score = sum(t.score for t in tasks)
        total_tasks = len(tasks)
        user_status = "Selected" if score == total_tasks and total_tasks > 0 else "Rejected"

        result.append({
            "id": u.id,
            "name": u.full_name,
            "college_id": u.college_id,
            "college_name": u.college_name,
            "mobile_number": u.mobile_number,
            "email": u.email,
            "score": score if total_tasks > 0 else "---",
            "status": user_status
        })

    if status:
        result = [r for r in result if r["status"].lower() == status.lower()]

    per_page = 14
    total_pages = (len(result) + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    paginated = result[start:end]

    return {
        "users": paginated,
        "total_pages": total_pages,
        "total_count": len(result)
    }

@app.get("/admin/users/export")
async def export_users_csv(_admin=Depends(require_admin), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User))
    users = res.scalars().all()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Email", "College", "Mobile", "Score", "Status"])
    for u in users:
        res_tasks = await db.execute(select(FileAssignment).filter(FileAssignment.user_id == u.id))
        tasks = res_tasks.scalars().all()
        score = sum(t.score for t in tasks)
        status = "Selected" if score == len(tasks) and len(tasks) > 0 else "Rejected"
        writer.writerow([u.full_name, u.email, u.college_name, u.mobile_number, score or "---", status])

    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=users.csv"
    })

@app.get("/admin/user/{user_id}/task/{index}")
async def get_user_task_detail(user_id: int, index: int, _admin=Depends(require_admin), db: AsyncSession = Depends(get_db)):
    res = await db.execute(
        select(FileAssignment)
        .filter(FileAssignment.user_id == user_id)
        .order_by(FileAssignment.id)
    )
    tasks = res.scalars().all()
    if not tasks or index >= len(tasks):
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[index]
    text = await read_text_from_s3(task.text_key)
    audio_url = get_presigned_url(task.audio_key)

    return {
        "question_no": index + 1,
        "audio_url": audio_url,
        "text": text,
        "user_answer": task.edited_text or task.action or "Not attempted",
        "status": "success"
    }
@app.get("/user/{user_id}/tasks")
async def get_user_tasks(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    try:
        res_user = await db.execute(
            select(User).filter(User.id == user_id)
        )
        user = res_user.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        res = await db.execute(
            select(FileAssignment)
            .filter(FileAssignment.user_id == user_id)
            .order_by(FileAssignment.id)
        )
        tasks = res.scalars().all()

        if not tasks:
            return {"tasks": []}

        sem = asyncio.Semaphore(S3_CONCURRENCY)

        async def safe_read_text(task):
            async with sem:
                try:
                    return await read_text_from_s3(task.text_key)
                except Exception:
                    logger.exception("Failed reading text %s", task.text_key)
                    return "[Error reading text]"

        async def safe_presign(task):
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, get_presigned_url, task.audio_key
                )
            except Exception:
                logger.exception("Failed presigning audio %s", task.audio_key)
                return None

        texts, urls = await asyncio.gather(
            asyncio.gather(*[safe_read_text(t) for t in tasks]),
            asyncio.gather(*[safe_presign(t) for t in tasks])
        )

        response = []
        for idx, task in enumerate(tasks, start=1):
            response.append({
                "id": task.id,
                "index": idx,
                "audio_url": urls[idx - 1],
                "text": texts[idx - 1],
                "user_answer": task.edited_text,
                "action": task.action or "Not attempted",
                "score": task.score,
                "completed": task.completed,
                "completed_at": task.completed_at
            })

        logger.info(
            "User tasks fetched | user_id=%s count=%d",
            user_id, len(response)
        )

        return {"tasks": response}

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Error fetching user tasks: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": "Failed to fetch tasks"
            }
        )

@app.get("/admin/overview")
async def admin_overview(
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):

    res = await db.execute(select(FileAssignment))
    tasks = res.scalars().all()

    assigned = len(tasks)
    corrected = sum(1 for t in tasks if t.action == "correct")
    incorrect = sum(1 for t in tasks if t.action == "incorrect")
    edited = sum(1 for t in tasks if t.action == "edit")

    s3_pairs = await list_pairs_s3()
    total_s3_tasks = len(s3_pairs)

    unassigned = max(total_s3_tasks - assigned, 0)

    return {
        "total_tasks": total_s3_tasks,
        "assigned_tasks": assigned,
        "unassigned_tasks": unassigned,
        "corrected_tasks": corrected,
        "incorrect_tasks": incorrect,
        "edited_tasks": edited
    }

@app.delete("/admin/user/{user_id}")
async def delete_user(
    user_id: int,
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    try:
        res = await db.execute(
            select(User).filter(User.id == user_id)
        )
        user = res.scalars().first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        await db.delete(user)
        await db.commit()

        logger.info("Admin deleted user %s and all assignments", user_id)

        return {
            "status": "success",
            "message": f"User {user_id} and all assignments deleted successfully"
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Failed to delete user %s: %s", user_id, exc)
        await db.rollback()
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": "Failed to fetch tasks; check server logs"
                }
        )
    
@app.post("/admin/tasks/assign")
async def assign_tasks_to_user(
    payload: AssignTasksIn,
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):

    res = await db.execute(select(User).filter(User.id == payload.user_id))
    user = res.scalars().first()
    if not user:
        raise HTTPException(404, "User not found")

    unassigned = await get_unassigned_tasks(db)
    if not unassigned:
        return {"message": "No unassigned tasks available"}

    tasks_to_assign = (
        unassigned[:payload.limit]
        if payload.limit
        else unassigned
    )

    for t in tasks_to_assign:
        db.add(FileAssignment(
            user_id=user.id,
            base=t["base"],
            audio_key=t["audio_key"],
            text_key=t["text_key"],
            json_key=t["json_key"]
        ))

    await db.commit()

    return {
        "message": "Tasks assigned successfully",
        "user_id": user.id,
        "assigned_count": len(tasks_to_assign)
}