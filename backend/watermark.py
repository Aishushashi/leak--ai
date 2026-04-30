import cv2
import numpy as np
import hashlib
from imwatermark import WatermarkEncoder, WatermarkDecoder

def embed_watermark(image_path, user_id):
    """
    Hides user_id invisibly inside the image.
    Uses DWT-DCT method — survives compression,
    screenshot, and minor edits.
    Cannot be seen by human eyes.
    """

    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # ── The watermark must be exactly 5 characters ──
    # We take first 5 chars of user_id
    # Example: "0xABC123..." becomes "0xABC"
    watermark_bytes = user_id[:5].encode('utf-8')

    # Embed the watermark invisibly
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', watermark_bytes)
    watermarked_img = encoder.encode(img, 'dwtDct')

    # Save the watermarked image
    filename    = image_path.split("/")[-1]
    output_path = f"uploads/watermarked_{filename}"
    cv2.imwrite(output_path, watermarked_img)

    print(f"✅ Invisible watermark embedded for user: {user_id[:5]}")
    return output_path


def extract_watermark(image_path):
    """
    Extracts the invisible watermark from an image.
    Returns the hidden user ID.
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Extract hidden watermark
    # 40 = number of bits (5 characters x 8 bits)
    decoder   = WatermarkDecoder('bytes', 40)
    watermark = decoder.decode(img, 'dwtDct')

    extracted = watermark.decode('utf-8', errors='ignore').strip()

    print(f"🔍 Extracted watermark: {extracted}")
    return extracted


def get_file_hash(file_path):
    """
    Generates unique fingerprint of the file.
    Same file = same hash always.
    """
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()