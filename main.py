import os
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
    frame_dir = f"frames/{video_id}/"
    extract_frames(video_path, frame_dir)

    # Load FAISS index + product IDs
    index = faiss.read_index("data/faiss_index.bin")
    with open("data/product_ids.txt") as f:
        product_ids = f.read().splitlines()

    products = []
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

            products.append({
                "type": det["class"],
                "match_type": match_type,
                "matched_product_id": prod_id,
                "confidence": round(score, 3)
            })

    vibes = classify_vibe(caption)

    result = {
        "video_id": video_id,
        "vibes": vibes,
        "products": products
    }
    return result

if __name__ == "__main__":
    # Test
    result = run_pipeline(
        video_id="test01",
        video_path="data/test_video.mp4",
        caption="Feeling super Y2K and coquette with this new dress 💅✨"
    )
    import json
    with open("outputs/test01.json", "w") as f:
        json.dump(result, f, indent=2)
