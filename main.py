from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import hashlib
from ai_detector import ai_compare_images
from fingerprint import extract_fingerprint, compare_fingerprints, save_fingerprint_to_db, search_database, get_database_stats

# ── Create app ────────────────────────────────────────
app = FastAPI(title="LeakGuard AI — AI Fingerprinting Backend")

# ── CORS ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Create uploads folder ─────────────────────────────
os.makedirs("uploads", exist_ok=True)

def get_file_hash(file_path):
    """Generates unique fingerprint of the file."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

# ════════════════════════════════════════════════
# ENDPOINT 1 — Home
# ════════════════════════════════════════════════
@app.get("/")
def home():
    return {
        "message": "LeakGuard AI — AI Fingerprinting Backend is running!",
        "features": [
            "AI Image Similarity",
            "Advanced Video Fingerprinting (pHash + AI)",
            "Local Fingerprint Database"
        ]
    }


# ════════════════════════════════════════════════
# ENDPOINT 2 — Register Video (Fingerprint)
# ════════════════════════════════════════════════
@app.post("/register/")
async def register_video(
    file:   UploadFile = File(...),
    owner:  str        = Form(...),
    title:  str        = Form(...)
):
    try:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Extract fingerprint
        fingerprint = extract_fingerprint(file_path)

        # 2. Save to local DB
        entry = save_fingerprint_to_db(fingerprint, owner, title)

        return {
            "status":  "success",
            "message": "Video registered successfully!",
            "id":      entry["id"],
            "title":   entry["title"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ════════════════════════════════════════════════
# ENDPOINT 3 — Search for Leak (Fingerprint Search)
# ════════════════════════════════════════════════
@app.post("/search/")
async def search_leak(file: UploadFile = File(...)):
    try:
        file_path = f"uploads/suspect_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Extract fingerprint from suspect video
        suspect_fp = extract_fingerprint(file_path)

        # 2. Search database
        result = search_database(suspect_fp)

        return {
            "status": "success",
            **result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ════════════════════════════════════════════════
# ENDPOINT 4 — Image Similarity (AI)
# ════════════════════════════════════════════════
@app.post("/similarity/")
async def image_similarity(
    original: UploadFile = File(...),
    suspect:  UploadFile = File(...)
):
    try:
        orig_path    = f"uploads/orig_{original.filename}"
        suspect_path = f"uploads/susp_{suspect.filename}"

        with open(orig_path, "wb") as f:
            shutil.copyfileobj(original.file, f)
        with open(suspect_path, "wb") as f:
            shutil.copyfileobj(suspect.file, f)

        result = ai_compare_images(orig_path, suspect_path)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ════════════════════════════════════════════════
# ENDPOINT 5 — Database Stats
# ════════════════════════════════════════════════
@app.get("/database/")
def database_stats():
    return get_database_stats()