import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import json
import os

# ════════════════════════════════════════════════
# Load pretrained AI model (ResNet)
# ════════════════════════════════════════════════

# Load model once when server starts
print("🤖 Loading AI model...")
ai_model = models.resnet50(pretrained=True)
ai_model.eval()
print("✅ AI model loaded!")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_ai_features(image_path):
    """Converts image to AI feature vector."""
    try:
        img    = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            features = ai_model(tensor)

        return features[0].numpy()
    except Exception as e:
        print(f"AI feature extraction failed: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """Compares two AI feature vectors."""
    dot_product = np.dot(vec1, vec2)
    magnitude   = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if magnitude == 0: return 0
    return dot_product / magnitude


def ai_compare_images(original_path, suspect_path):
    """Uses DEEP LEARNING to compare two images."""
    print("🤖 Running AI deep comparison...")
    features1 = get_ai_features(original_path)
    features2 = get_ai_features(suspect_path)

    if features1 is None or features2 is None:
        return {"status": "error", "message": "Could not extract AI features"}

    similarity     = cosine_similarity(features1, features2)
    similarity_pct = round(similarity * 100, 2)
    is_leak        = similarity_pct > 75

    return {
        "aiSimilarityPercent": similarity_pct,
        "isLeak":              is_leak,
        "confidence":          "HIGH" if similarity_pct > 90 else "MEDIUM" if similarity_pct > 75 else "LOW",
        "verdict":             "🚨 LEAK DETECTED" if is_leak else "✅ Not same content",
        "method":              "Deep Learning (ResNet50)"
    }


def ai_compare_videos(original_path, suspect_path):
    """Uses AI to compare two videos frame by frame."""
    print("🤖 Running AI video comparison...")
    cap1         = cv2.VideoCapture(original_path)
    cap2         = cv2.VideoCapture(suspect_path)
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_points = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4, total_frames - 1]
    similarities  = []

    for i, point in enumerate(sample_points):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, point)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, point)
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 and ret2:
            path1 = f"uploads/temp_f1_{i}.jpg"
            path2 = f"uploads/temp_f2_{i}.jpg"
            cv2.imwrite(path1, frame1)
            cv2.imwrite(path2, frame2)

            f1 = get_ai_features(path1)
            f2 = get_ai_features(path2)

            if f1 is not None and f2 is not None:
                sim = cosine_similarity(f1, f2)
                similarities.append(round(sim * 100, 2))

            os.remove(path1)
            os.remove(path2)

    cap1.release()
    cap2.release()

    if not similarities:
        return {"aiSimilarityPercent": 0, "isLeak": False, "verdict": "Could not compare videos"}

    avg_similarity = round(sum(similarities) / len(similarities), 2)
    is_leak        = avg_similarity > 75

    return {
        "aiSimilarityPercent": avg_similarity,
        "isLeak":              is_leak,
        "verdict":             "🚨 VIDEO LEAK DETECTED" if is_leak else "✅ Not same content",
        "method":              "Deep Learning (ResNet50)"
    }


def generate_leak_report(similarity_result, info_text, video_path):
    """AI generates a complete leak report automatically."""
    similarity = similarity_result.get("aiSimilarityPercent", 0)
    is_leak    = similarity_result.get("isLeak", False)

    if similarity > 95: severity, action = "CRITICAL", "Immediate legal action recommended"
    elif similarity > 85: severity, action = "HIGH", "File DMCA takedown immediately"
    elif similarity > 75: severity, action = "MEDIUM", "Monitor and gather more evidence"
    else: severity, action = "LOW", "No immediate action needed"

    report = {
        "reportTitle":    "LeakGuard AI — AI Detection Report",
        "verdict":        "LEAK CONFIRMED" if is_leak else "NO LEAK FOUND",
        "severity":       severity if is_leak else "NONE",
        "similarityScore": f"{similarity}%",
        "status":         info_text,
        "actionRequired": action if is_leak else "None",
        "evidence": {
            "aiSimilarity": f"{similarity}%",
            "fileAnalyzed":  os.path.basename(video_path)
        }
    }

    report_path = f"uploads/leak_report_{os.path.basename(video_path)}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report