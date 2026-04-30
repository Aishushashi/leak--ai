from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from watermark import embed_watermark, get_file_hash
from detect import extract_watermark, check_similarity
from video_watermark import embed_watermark_video, extract_watermark_video, check_video_similarity
from blockchain import get_wallet_details, register_content_on_blockchain

# ── Create app ────────────────────────────────────────
app = FastAPI(title="LeakGuard AI Backend")

# ── CORS ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Create uploads folder ─────────────────────────────
os.makedirs("uploads", exist_ok=True)


# ════════════════════════════════════════════════
# ENDPOINT 1 — Check if server is running
# ════════════════════════════════════════════════
@app.get("/")
def home():
    return {
        "message": "LeakGuard AI Backend is running!",
        "endpoints": [
            "POST /upload/              → embed watermark in image",
            "POST /detect/             → extract watermark from image",
            "POST /similarity/         → check image similarity",
            "POST /leak/full/          → full image leak detection",
            "POST /leak/identify/      → identify leaker wallet",
            "POST /blockchain/register/→ register on blockchain",
            "POST /video/upload/       → embed watermark in video",
            "POST /video/detect/       → detect leak in video"
        ]
    }


# ════════════════════════════════════════════════
# ENDPOINT 2 — Upload image + embed watermark
# ════════════════════════════════════════════════
@app.post("/upload/")
async def upload_file(
    file:   UploadFile = File(...),
    userId: str        = Form(...)
):
    try:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_path = embed_watermark(file_path, userId)
        file_hash   = get_file_hash(file_path)

        return {
            "status":          "success",
            "message":         "Watermark embedded!",
            "watermarkedFile": output_path,
            "fileHash":        file_hash,
            "embeddedUserId":  userId
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ════════════════════════════════════════════════
# ENDPOINT 3 — Extract watermark from image
# ════════════════════════════════════════════════
@app.post("/detect/")
async def detect_file(file: UploadFile = File(...)):
    try:
        file_path = f"uploads/suspect_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_id = extract_watermark(file_path)

        return {
            "status":      "success",
            "extractedId": extracted_id,
            "message":     f"Watermark found: {extracted_id}"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ════════════════════════════════════════════════
# ENDPOINT 4 — Check similarity between 2 images
# ════════════════════════════════════════════════
@app.post("/similarity/")
async def similarity_check(
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

        result = check_similarity(orig_path, suspect_path)
        return {"status": "success", **result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ════════════════════════════════════════════════
# ENDPOINT 5 — Full image leak detection
# ════════════════════════════════════════════════
@app.post("/leak/full/")
async def full_leak_detection(
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

        similarity_result = check_similarity(orig_path, suspect_path)
        extracted_id      = extract_watermark(suspect_path)
        file_hash         = get_file_hash(suspect_path)

        return {
            "status":            "success",
            "similarityPercent": similarity_result["similarityPercent"],
            "isLeak":            similarity_result["isLeak"],
            "verdict":           similarity_result["verdict"],
            "extractedUserId":   extracted_id,
            "suspectFileHash":   file_hash
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ════════════════════════════════════════════════
# ENDPOINT 6 — Identify leaker wallet
# ════════════════════════════════════════════════
@app.post("/leak/identify/")
async def identify_leaker(file: UploadFile = File(...)):
    try:
        file_path = f"uploads/suspect_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 1 — extract watermark
        extracted_wallet = extract_watermark(file_path)

        if extracted_wallet == "No watermark found":
            return {
                "status":  "no_watermark",
                "message": "No watermark found in this file"
            }

        # Step 2 — get wallet details from blockchain
        wallet_details = get_wallet_details(extracted_wallet)

        # Step 3 — hash of leaked file
        file_hash = get_file_hash(file_path)

        return {
            "status":         "success",
            "verdict":        "LEAKER IDENTIFIED",
            "leakerWallet":   extracted_wallet,
            "walletDetails":  wallet_details,
            "leakedFileHash": file_hash,
            "evidence": {
                "extractedFrom":  file_path,
                "watermarkFound": extracted_wallet,
                "explorerLink":   wallet_details.get("explorerLink")
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ════════════════════════════════════════════════
# ENDPOINT 7 — Register content on blockchain
# ════════════════════════════════════════════════
@app.post("/blockchain/register/")
async def register_content(
    file:   UploadFile = File(...),
    userId: str        = Form(...),
    title:  str        = Form(...)
):
    try:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_hash = get_file_hash(file_path)
        result    = register_content_on_blockchain(userId, file_hash, title)

        return {
            "status":     "success",
            "fileHash":   file_hash,
            "blockchain": result
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ════════════════════════════════════════════════
# ENDPOINT 8 — Upload video + embed watermark
# ════════════════════════════════════════════════
@app.post("/video/upload/")
async def upload_video(
    file:   UploadFile = File(...),
    userId: str        = Form(...)
):
    try:
        allowed = ['.mp4', '.avi', '.mov', '.mkv']
        ext     = os.path.splitext(file.filename)[1].lower()

        if ext not in allowed:
            return {
                "status":  "error",
                "message": f"Only video files allowed: {allowed}"
            }

        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_path = embed_watermark_video(file_path, userId)
        file_hash   = get_file_hash(file_path)

        return {
            "status":          "success",
            "message":         "Video watermarked successfully!",
            "watermarkedFile": output_path,
            "fileHash":        file_hash,
            "embeddedUserId":  userId
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ════════════════════════════════════════════════
# ENDPOINT 9 — Detect leak in video
# ════════════════════════════════════════════════
@app.post("/video/detect/")
async def detect_video_leak(
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

        similarity   = check_video_similarity(orig_path, suspect_path)
        extracted_id = extract_watermark_video(suspect_path)

        wallet_details = {}
        if similarity["isLeak"] and extracted_id != "No watermark found":
            wallet_details = get_wallet_details(extracted_id)

        return {
            "status":           "success",
            "similarityResult": similarity,
            "extractedUserId":  extracted_id,
            "leakerDetails":    wallet_details,
            "verdict":          "VIDEO LEAK DETECTED" if similarity["isLeak"] else "No leak found"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}