import imagehash
from PIL import Image
from watermark import extract_watermark

def check_similarity(original_path, suspect_path):
    """
    Compares two images and returns how similar they are.
    Above 85% = likely the same content = likely a leak.
    """
    try:
        # Open both images
        img1 = Image.open(original_path)
        img2 = Image.open(suspect_path)

        # Calculate perceptual hash of both
        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)

        # Hash distance — 0 means identical, 64 means completely different
        distance = hash1 - hash2

        # Convert to percentage (better formula)
        similarity = round((1 - distance / 64) * 100, 2)
        is_leak    = similarity > 85

        return {
            "similarityPercent": similarity,
            "isLeak":            is_leak,
            "verdict":           "⚠️ LEAK DETECTED" if is_leak else "✅ Not same content",
            "hashDistance":      distance   # shows raw distance for debugging
        }

    except Exception as e:
        return {"error": str(e)}
def full_detection(original_path, suspect_path):
    """
    Does everything in one go:
    1. Checks similarity
    2. Extracts watermark
    Returns full leak report.
    """
    # Step 1 — similarity
    similarity_result = check_similarity(original_path, suspect_path)

    # Step 2 — extract watermark from suspected file
    try:
        extracted_id = extract_watermark(suspect_path)
    except Exception:
        extracted_id = "Could not extract watermark"

    return {
        "similarityPercent": similarity_result.get("similarityPercent", 0),
        "isLeak":            similarity_result.get("isLeak", False),
        "verdict":           similarity_result.get("verdict", "Unknown"),
        "extractedUserId":   extracted_id
    }