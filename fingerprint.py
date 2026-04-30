import cv2
import numpy as np
import hashlib
import json
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import imagehash
from PIL import Image
from datetime import datetime


# ════════════════════════════════════════════════
# LOAD AI MODEL (ResNet)
# Loads once when server starts
# Used for deep feature extraction
# ════════════════════════════════════════════════
print("🤖 Loading AI model for fingerprinting...")

ai_model = models.resnet50(pretrained=True)

# Remove last layer → gives us features not classification
ai_model = torch.nn.Sequential(
    *list(ai_model.children())[:-1]
)
ai_model.eval()
print("✅ AI Model loaded!")

# Image transform for AI model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


# ════════════════════════════════════════════════
# METHOD 1 — PERCEPTUAL HASH (pHash)
# Fast, lightweight
# Good for exact/near-exact copies
# ════════════════════════════════════════════════

def get_phash(frame: np.ndarray) -> str:
    """
    Converts a video frame to perceptual hash.

    How it works simply:
    1. Resize frame to 32x32
    2. Convert to grayscale
    3. Apply DCT (frequency transform)
    4. Compare values to median
    5. Output 64-bit hash string

    Similar images → similar hashes!
    Even if brightness/color changed slightly
    """
    # Convert OpenCV BGR to PIL RGB
    pil_image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    # Get perceptual hash
    phash = imagehash.phash(pil_image)
    return str(phash)


def compare_phash(hash1: str, hash2: str) -> float:
    """
    Compares two perceptual hashes.
    Returns similarity from 0 to 100%.
    0  = completely different
    100 = identical
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)

    distance   = h1 - h2       # 0 to 64
    similarity = (1 - distance / 64) * 100
    return round(similarity, 2)


# ════════════════════════════════════════════════
# METHOD 2 — AI DEEP FEATURES (PyTorch ResNet)
# Slower but smarter
# Works even if video is cropped/rotated/recolored
# ════════════════════════════════════════════════

def get_ai_features(frame: np.ndarray) -> list:
    """
    Converts frame to AI feature vector using ResNet.

    How it works simply:
    1. Feed frame to ResNet AI model
    2. Model outputs 2048 numbers
    3. These numbers describe the IMAGE CONTENT
    4. Similar images → similar numbers!

    Unlike pHash, this understands CONTENT:
    - pHash fails if video is slightly cropped
    - AI features still match!
    """
    # Convert to PIL
    pil_image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    # Transform for model
    tensor = transform(pil_image).unsqueeze(0)

    # Get features
    with torch.no_grad():
        features = ai_model(tensor)

    # Flatten to 1D list
    feature_vector = features.squeeze().numpy().tolist()
    return feature_vector


def compare_ai_features(
    features1: list,
    features2: list
) -> float:
    """
    Compares two AI feature vectors.
    Uses cosine similarity.
    Returns 0 to 100%.
    """
    v1  = np.array(features1)
    v2  = np.array(features2)

    dot = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)

    if mag == 0:
        return 0.0

    similarity = (dot / mag) * 100
    return round(float(similarity), 2)


# ════════════════════════════════════════════════
# CORE — Extract Video Fingerprint
# Combines both pHash + AI features
# ════════════════════════════════════════════════

def extract_fingerprint(
    video_path:          str,
    sample_every_seconds: int  = 2,
    use_ai:              bool = True
) -> dict:
    """
    Extracts complete fingerprint from a video.

    Process:
    1. Open video
    2. Pick 1 frame every N seconds (not all frames!)
    3. For each picked frame:
       a. Generate pHash (fast method)
       b. Generate AI features (smart method)
    4. Store all hashes → tiny fingerprint!

    Result:
    2hr movie = ~3600 frame hashes
    Size = ~500KB (vs 50GB for full video!)
    """
    print(f"\n🎬 Extracting fingerprint from: {video_path}")
    print("━" * 50)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # ── Get video info ────────────────────────────
    fps           = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_secs = total_frames / fps if fps > 0 else 0

    print(f"📊 Video Info:")
    print(f"   Resolution : {width}x{height}")
    print(f"   FPS        : {fps:.1f}")
    print(f"   Duration   : {duration_secs:.1f} seconds")
    print(f"   Total frames: {total_frames}")

    # ── Calculate sampling ─────────────────────────
    frame_interval = max(1, int(fps * sample_every_seconds))
    expected_samples = total_frames // frame_interval

    print(f"\n📸 Sampling 1 frame every {sample_every_seconds} seconds")
    print(f"   Will sample ~{expected_samples} frames")
    print(f"   (Instead of all {total_frames} frames!)")
    print("\n⏳ Processing frames...")

    # ── Extract frames and generate hashes ─────────
    frame_fingerprints = []
    frame_number       = 0
    sampled_count      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every Nth frame
        if frame_number % frame_interval == 0:
            timestamp = round(frame_number / fps, 2)

            # Method 1: pHash (always)
            phash = get_phash(frame)

            # Method 2: AI features (optional, slower)
            ai_features = None
            if use_ai and sampled_count % 10 == 0:
                # AI on every 10th sampled frame
                # Balances accuracy vs speed
                ai_features = get_ai_features(frame)

            frame_fingerprints.append({
                "frameNumber": frame_number,
                "timestamp":   timestamp,
                "pHash":       phash,
                "aiFeatures":  ai_features
            })

            sampled_count += 1

            # Progress update every 50 samples
            if sampled_count % 50 == 0:
                print(f"   Processed {sampled_count} frames...")

        frame_number += 1

    cap.release()

    # ── Generate master hash ───────────────────────
    # Combine all pHashes into one master hash
    all_hashes  = "".join([f["pHash"] for f in frame_fingerprints])
    master_hash = hashlib.sha256(all_hashes.encode()).hexdigest()

    # ── Calculate fingerprint size ─────────────────
    fp_size_bytes = len(json.dumps(frame_fingerprints).encode())
    fp_size_kb    = round(fp_size_bytes / 1024, 2)

    print(f"\n✅ Fingerprint extracted!")
    print(f"   Sampled frames : {sampled_count}")
    print(f"   Master hash    : {master_hash[:20]}...")
    print(f"   Fingerprint size: {fp_size_kb} KB")
    print(f"   (Original video would be MB/GB!)")

    return {
        "masterHash":        master_hash,
        "videoPath":         video_path,
        "videoInfo": {
            "fps":           fps,
            "totalFrames":   total_frames,
            "durationSecs":  duration_secs,
            "resolution":    f"{width}x{height}"
        },
        "samplingInfo": {
            "sampleEvery":   sample_every_seconds,
            "sampledFrames": sampled_count,
            "totalFrames":   total_frames,
            "compressionRatio": f"{total_frames}:{sampled_count}"
        },
        "fingerprintData":   frame_fingerprints,
        "fingerprintSizeKB": fp_size_kb,
        "extractedAt":       datetime.now().isoformat()
    }


# ════════════════════════════════════════════════
# CORE — Compare Two Fingerprints
# ════════════════════════════════════════════════

def compare_fingerprints(
    fp1:       dict,
    fp2:       dict,
    threshold: float = 85.0
) -> dict:
    """
    Compares two video fingerprints.

    Uses both methods:
    1. pHash comparison (fast, every frame)
    2. AI feature comparison (smart, key frames)

    Final score = weighted average of both

    Works even if:
    ✅ Video is re-encoded
    ✅ Resolution changed
    ✅ Slight brightness/color change
    ✅ Recorded on phone (screen recording)
    """
    print(f"\n🔍 Comparing fingerprints...")
    print("━" * 50)

    hashes1 = fp1.get("fingerprintData", [])
    hashes2 = fp2.get("fingerprintData", [])

    if not hashes1 or not hashes2:
        return {
            "similarityPercent": 0,
            "isMatch":           False,
            "verdict":           "Could not compare — empty fingerprint"
        }

    # ── pHash comparison ──────────────────────────
    min_length     = min(len(hashes1), len(hashes2))
    phash_scores   = []

    for i in range(min_length):
        score = compare_phash(
            hashes1[i]["pHash"],
            hashes2[i]["pHash"]
        )
        phash_scores.append(score)

    avg_phash = round(
        sum(phash_scores) / len(phash_scores), 2
    ) if phash_scores else 0

    # ── AI feature comparison ─────────────────────
    ai_scores = []

    for i in range(min_length):
        f1 = hashes1[i].get("aiFeatures")
        f2 = hashes2[i].get("aiFeatures")
        if f1 and f2:
            score = compare_ai_features(f1, f2)
            ai_scores.append(score)

    avg_ai = round(
        sum(ai_scores) / len(ai_scores), 2
    ) if ai_scores else avg_phash

    # ── Combined score (weighted) ─────────────────
    # pHash = 60% weight (fast, reliable for exact copies)
    # AI    = 40% weight (smart, for edited copies)
    final_score = round(
        (avg_phash * 0.6) + (avg_ai * 0.4), 2
    )

    is_match = final_score >= threshold

    # ── Verdict ───────────────────────────────────
    if final_score >= 95:
        verdict    = "🚨 EXACT COPY DETECTED"
        confidence = "VERY HIGH"
    elif final_score >= 85:
        verdict    = "🚨 LIKELY PIRATED COPY"
        confidence = "HIGH"
    elif final_score >= 70:
        verdict    = "⚠️ SUSPICIOUS — Possible edit"
        confidence = "MEDIUM"
    else:
        verdict    = "✅ Different content"
        confidence = "HIGH"

    print(f"\n📊 Comparison Results:")
    print(f"   pHash similarity  : {avg_phash}%")
    print(f"   AI similarity     : {avg_ai}%")
    print(f"   Final score       : {final_score}%")
    print(f"   Verdict           : {verdict}")

    return {
        "finalSimilarity":  final_score,
        "pHashSimilarity":  avg_phash,
        "aiSimilarity":     avg_ai,
        "isMatch":          is_match,
        "confidence":       confidence,
        "verdict":          verdict,
        "framesCompared":   min_length,
        "threshold":        threshold
    }


# ════════════════════════════════════════════════
# DATABASE — Save Fingerprint
# ════════════════════════════════════════════════

def save_fingerprint_to_db(
    fingerprint:   dict,
    owner_wallet:  str,
    title:         str,
    content_type:  str = "video"
) -> dict:
    """
    Saves fingerprint to local database.
    In production → MongoDB Atlas.

    Note: We save ONLY the fingerprint
    NOT the actual video!
    """
    db_path = "uploads/fingerprint_db.json"

    # Load existing database
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            db = json.load(f)
    else:
        db = []

    # Create entry
    entry = {
        "id":           hashlib.sha256(
                            fingerprint["masterHash"].encode()
                        ).hexdigest()[:12],
        "owner":        owner_wallet,
        "title":        title,
        "contentType":  content_type,
        "masterHash":   fingerprint["masterHash"],
        "videoInfo":    fingerprint["videoInfo"],
        "samplingInfo": fingerprint["samplingInfo"],
        "fingerprint":  fingerprint,
        "registeredAt": datetime.now().isoformat()
    }

    db.append(entry)

    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)

    print(f"\n💾 Fingerprint saved to database!")
    print(f"   ID    : {entry['id']}")
    print(f"   Owner : {owner_wallet}")
    print(f"   Title : {title}")
    print(f"   Size  : {fingerprint['fingerprintSizeKB']} KB")

    return entry


# ════════════════════════════════════════════════
# DATABASE — Search Fingerprint Database
# ════════════════════════════════════════════════

def search_database(
    suspect_fingerprint: dict,
    threshold:           float = 85.0
) -> dict:
    """
    Searches entire fingerprint database
    for a matching video.

    Process:
    1. Extract fingerprint from suspect video
    2. Compare against ALL registered fingerprints
    3. Return match if found

    Can search 1000s of videos in seconds!
    No need to upload original video!
    """
    db_path = "uploads/fingerprint_db.json"

    if not os.path.exists(db_path):
        return {
            "found":   False,
            "message": "Database empty — no videos registered yet"
        }

    with open(db_path, 'r') as f:
        db = json.load(f)

    if not db:
        return {
            "found":   False,
            "message": "No videos in database"
        }

    print(f"\n🔍 Searching {len(db)} registered videos...")
    print("━" * 50)

    best_match  = None
    best_score  = 0

    for entry in db:
        print(f"\n   Checking: {entry['title']}")

        result = compare_fingerprints(
            entry["fingerprint"],
            suspect_fingerprint,
            threshold
        )

        if result["finalSimilarity"] > best_score:
            best_score = result["finalSimilarity"]
            best_match = {
                "entry":  entry,
                "result": result
            }

        # If very high match found stop searching
        if result["finalSimilarity"] >= 95:
            print(f"   🚨 Very high match found! Stopping search.")
            break

    if best_match and best_match["result"]["isMatch"]:
        return {
            "found":           True,
            "matchedTitle":    best_match["entry"]["title"],
            "matchedOwner":    best_match["entry"]["owner"],
            "matchedId":       best_match["entry"]["id"],
            "registeredAt":    best_match["entry"]["registeredAt"],
            "similarityResult": best_match["result"],
            "verdict":         "🚨 PIRATED CONTENT DETECTED",
            "evidence": {
                "masterHash":    best_match["entry"]["masterHash"],
                "originalOwner": best_match["entry"]["owner"],
                "originalTitle": best_match["entry"]["title"]
            }
        }

    return {
        "found":         False,
        "bestScore":     best_score,
        "verdict":       "✅ No match found in database",
        "videosChecked": len(db)
    }


# ════════════════════════════════════════════════
# UTILITY — Get database stats
# ════════════════════════════════════════════════

def get_database_stats() -> dict:
    """Shows what's in the fingerprint database."""
    db_path = "uploads/fingerprint_db.json"

    if not os.path.exists(db_path):
        return {
            "totalVideos":     0,
            "totalOwners":     0,
            "message":         "Database is empty"
        }

    with open(db_path, 'r') as f:
        db = json.load(f)

    owners = list(set([e["owner"] for e in db]))

    return {
        "totalVideos":     len(db),
        "totalOwners":     len(owners),
        "registeredVideos": [
            {
                "id":    e["id"],
                "title": e["title"],
                "owner": e["owner"],
                "date":  e["registeredAt"]
            }
            for e in db
        ]
    }