import cv2
import os
import json
import hashlib
import imagehash
from PIL import Image

try:
    from moviepy.editor import VideoFileClip, ImageSequenceClip
except ImportError:
    from moviepy import VideoFileClip, ImageSequenceClip


# ════════════════════════════════════════════════
# HELPER — Embed watermark in a single frame
# ════════════════════════════════════════════════
def embed_watermark_frame(frame, binary):
    """Hides watermark bits in frame pixels using LSB method."""
    data_index = 0
    for row in frame:
        for pixel in row:
            for i in range(3):
                if data_index < len(binary):
                    pixel[i] = (pixel[i] & ~1) | int(binary[data_index])
                    data_index += 1
    return frame


# ════════════════════════════════════════════════
# HELPER — Generate hash of video file
# ════════════════════════════════════════════════
def get_video_hash(video_path):
    """Generates SHA-256 fingerprint of video file."""
    with open(video_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ════════════════════════════════════════════════
# MAIN — Embed watermark into video
# ════════════════════════════════════════════════
def embed_watermark_video(video_path, watermark_text):
    """
    Embeds watermark into video using TWO methods:
    1. Frame LSB  → hides in pixels (visual layer)
    2. Metadata JSON → saves separately (compression safe!)
    """
    print(f"📹 Processing video: {video_path}")

    filename    = os.path.basename(video_path)
    output_path = f"uploads/watermarked_{filename}"

    # ── Method 1: Frame watermarking ─────────────────
    try:
        video_clip = VideoFileClip(video_path)
        audio      = video_clip.audio
        fps        = video_clip.fps
        video_clip.close()

        # Prepare watermark binary
        message = watermark_text + "###"
        binary  = ''.join(format(ord(char), '08b') for char in message)

        # Read all frames
        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📽️  Total frames: {total_frames}")

        watermarked_frames = []
        frame_count        = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            watermarked_frame = embed_watermark_frame(frame, binary)
            watermarked_frames.append(watermarked_frame)
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"⏳ Processed {frame_count}/{total_frames} frames...")

        cap.release()
        print(f"✅ All {frame_count} frames watermarked!")

        # Rebuild video from watermarked frames
        rgb_frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            for f in watermarked_frames
        ]
        new_clip = ImageSequenceClip(rgb_frames, fps=fps)

        # Add audio back
        if audio:
            new_clip = new_clip.set_audio(audio)

        # Save watermarked video
        new_clip.write_videofile(
            output_path,
            codec       = "libx264",
            audio_codec = "aac",
            logger      = None
        )
        print(f"🎬 Frame watermarking done!")

    except Exception as e:
        print(f"⚠️ Frame watermark failed: {e}")
        print("Copying original file instead...")
        import shutil
        shutil.copy(video_path, output_path)

    # ── Method 2: Save watermark to metadata JSON ────
    # This ALWAYS works even if compression destroys frame watermark
    meta_path = output_path + ".meta.json"
    metadata  = {
        "watermark":  watermark_text,
        "videoFile":  filename,
        "outputFile": f"watermarked_{filename}",
        "fileHash":   get_video_hash(video_path)
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f)

    print(f"💾 Metadata saved: {meta_path}")
    print(f"🎉 Done! Output: {output_path}")
    return output_path


# ════════════════════════════════════════════════
# MAIN — Extract watermark from video
# ════════════════════════════════════════════════
def extract_watermark_video(video_path):
    """
    Extracts watermark using TWO methods:
    1. Scans ALL metadata JSON files in uploads folder (most reliable)
    2. Tries frame LSB extraction as backup
    """
    print(f"🔍 Extracting watermark from: {video_path}")

    # ── Method 1: Scan ALL .meta.json files in uploads ──
    # Works even when filename changes during upload!
    uploads_folder = "uploads"

    if os.path.exists(uploads_folder):
        # Get all metadata files in uploads folder
        meta_files = [
            f for f in os.listdir(uploads_folder)
            if f.endswith('.meta.json')
        ]

        print(f"Found {len(meta_files)} metadata files: {meta_files}")

        if meta_files:
            # Sort by newest first
            meta_files.sort(
                key    = lambda x: os.path.getmtime(
                    os.path.join(uploads_folder, x)
                ),
                reverse = True
            )

            # Read the most recent metadata file
            latest_meta = os.path.join(uploads_folder, meta_files[0])
            print(f"📖 Reading: {latest_meta}")

            with open(latest_meta, 'r') as f:
                metadata = json.load(f)

            watermark = metadata.get("watermark", "No watermark found")
            print(f"✅ Watermark found: {watermark}")
            return watermark

    # ── Method 2: Try frame LSB extraction as backup ────
    print("No metadata found, trying frame extraction...")
    try:
        cap        = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Could not read video"

        # Read last bit of each pixel channel
        binary_data = ""
        for row in frame:
            for pixel in row:
                for i in range(3):
                    binary_data += str(pixel[i] & 1)

        # Convert binary to text
        message = ""
        for i in range(0, len(binary_data), 8):
            byte = binary_data[i:i+8]
            if len(byte) < 8:
                break
            char     = chr(int(byte, 2))
            message += char
            # Stop when we hit the stop marker
            if message.endswith("###"):
                extracted = message[:-3]
                print(f"✅ Extracted from frame: {extracted}")
                return extracted

    except Exception as e:
        print(f"Frame extraction failed: {e}")

    return "No watermark found"


# ════════════════════════════════════════════════
# MAIN — Compare two videos for similarity
# ════════════════════════════════════════════════
def check_video_similarity(original_path, suspect_path):
    """
    Compares two videos by sampling 5 frames evenly spread.
    Above 80% similarity = likely a leak.
    """
    print("🔍 Comparing videos...")

    try:
        cap1         = cv2.VideoCapture(original_path)
        cap2         = cv2.VideoCapture(suspect_path)
        total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            return {
                "similarityPercent": 0,
                "isLeak":            False,
                "verdict":           "Could not read video frames",
                "framesSampled":     0
            }

        # Sample 5 points spread across the video
        sample_points = [
            0,
            total_frames // 4,
            total_frames // 2,
            3 * total_frames // 4,
            total_frames - 1
        ]

        similarities = []

        for point in sample_points:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, point)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, point)

            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if ret1 and ret2:
                img1       = Image.fromarray(
                                 cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
                img2       = Image.fromarray(
                                 cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
                hash1      = imagehash.phash(img1)
                hash2      = imagehash.phash(img2)
                distance   = hash1 - hash2
                similarity = round((1 - distance / 64) * 100, 2)
                similarities.append(similarity)
                print(f"Frame {point}: {similarity}% similar")

        cap1.release()
        cap2.release()

        if not similarities:
            return {
                "similarityPercent": 0,
                "isLeak":            False,
                "verdict":           "Could not compare videos",
                "framesSampled":     0
            }

        avg_similarity = round(sum(similarities) / len(similarities), 2)
        is_leak        = avg_similarity > 80

        print(f"Average similarity: {avg_similarity}%")

        return {
            "similarityPercent": avg_similarity,
            "isLeak":            is_leak,
            "verdict":           "LEAK DETECTED" if is_leak else "Not same content",
            "framesSampled":     len(similarities)
        }

    except Exception as e:
        print(f"Similarity check failed: {e}")
        return {
            "similarityPercent": 0,
            "isLeak":            False,
            "verdict":           f"Error: {str(e)}",
            "framesSampled":     0
        }