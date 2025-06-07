def run_pipeline(video_id, video_path, caption):
    from models.yolo_detector import detect_fashion_items
    from utils.frame_extractor import extract_frames
    from models.clip_matcher import embed_image, find_best_match
    from models.vibe_classifier import classify_vibe
    import cv2
    import faiss
    from pathlib import Path
    from PIL import Image
    import json

    frame_dir = f"frames/{video_id}/"
    extract_frames(video_path, frame_dir)

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

            print(f"[DEBUG] Matching {det['class']} – Top product: {prod_id} (score: {round(score, 3)})")

            products.append({
                "type": det["class"],
                "match_type": match_type,
                "matched_product_id": prod_id,
                "confidence": round(score, 3)
            })

    vibes = classify_vibe(caption)
    print(f"[DEBUG] Vibes for caption '{caption}': {vibes}")

    result = {
        "video_id": video_id,
        "vibes": vibes,
        "products": products
    }
    return result
