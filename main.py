import os
import json
import cv2
from pathlib import Path
from PIL import Image
import faiss
import numpy as np

from utils.frame_extractor import extract_frames
from models.yolo_detector import detect_fashion_items
from models.clip_matcher import embed_image, find_best_match
from models.vibe_classifier import classify_vibe


def run_pipeline(video_id, video_path, caption):
    # Ensure output folders exist
    frame_dir = f"frames/{video_id}/"
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Step 1: Extract frames
    extract_frames(video_path, frame_dir)

    # Step 2: Load FAISS index and product IDs
    try:
        index = faiss.read_index("data/faiss_index.bin")
        with open("data/product_ids.txt") as f:
            product_ids = f.read().splitlines()
    except Exception as e:
        raise FileNotFoundError("Error loading FAISS index or product_ids.txt") from e

    products = []

    # Step 3: Detect fashion items in each frame
    os.makedirs("debug_crops", exist_ok=True)  # Create folder once

    for frame_file in Path(frame_dir).glob("*.jpg"):
        detections = detect_fashion_items(str(frame_file))
    img = cv2.imread(str(frame_file))

    for det in detections:
        x, y, w, h = map(int, det["bbox"])
        x1, y1 = max(x - w // 2, 0), max(y - h // 2, 0)
        x2, y2 = x1 + w, y1 + h
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_img = Image.fromarray(crop)
        emb = embed_image(crop_img)
        match_type, prod_id, score = find_best_match(emb, index, product_ids)

        print(f"Matching {det['class']} – Top product: {prod_id} (score: {score:.3f})")

        # Save the crop with class name for visual debugging
        crop_img.save(f"debug_crops/{video_id}_{frame_file.stem}_{det['class']}.jpg")

        if score < 0.75:
            continue

        products.append({
            "type": det["class"],
            "match_type": match_type,
            "matched_product_id": prod_id,
            "confidence": round(float(score), 3)
        })


    # Step 4: Classify vibes
    caption = caption if caption.strip() else "fashion style"
    vibes = classify_vibe(caption)

    # Step 5: Format result
    result = {
        "video_id": video_id,
        "vibes": vibes,
        "products": products
    }

    return result


if __name__ == "__main__":
    result = run_pipeline(
        video_id="test01",
        video_path="data/test_video.mp4",
        caption="Feeling super Y2K and coquette with this new dress"
    )

    with open("outputs/test01.json", "w") as f:
        json.dump(result, f, indent=2)
