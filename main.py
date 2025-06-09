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
    #step 1 - extract frames from video and save them in a dedicated directory 
    frame_dir = f"frames/{video_id}/"
    extract_frames(video_path, frame_dir)
    #step 2 - load the prebuilt faiss index containg product image embeddings 
    index = faiss.read_index("data/faiss_index.bin")
    #sep 3 - load the list of product IDS corresponding to the faiss index 
    with open("data/product_ids.txt") as f:
        product_ids = f.read().splitlines()
    #step4 = prepare to store all detected fashion items  and matched product into 
    products = []
    #Step 5 = loop over each extracted frame in the frame directory 
    for frame_file in Path(frame_dir).glob("*.jpg"):
        #detect fashion items in the current frame 
        detections = detect_fashion_items(str(frame_file))
        #read the frame image using OpenCV for cropping
        img = cv2.imread(str(frame_file))
        #step6 - loop over each detected item in the frame 
        for det in detections:
            #get the bounding box coordinates 
            x, y, w, h = map(int, det["bbox"])
            #conert the from center-based to top-left-based
            x1, y1 = max(x - w // 2, 0), max(y - h // 2, 0)
            x2, y2 = x1 + w, y1 + h
            #crop the detected item from the original image 
            crop = img[y1:y2, x1:x2]
            #skip empty or invalid 
            if crop.size == 0:
                continue
            #covert the cropped image to PIL format for CLIP embedding 
            crop_img = Image.fromarray(crop)
            #get the clip embedding of the cropped image 
            emb = embed_image(crop_img)
            #search the faiss index for the most similar product 
            match_type, prod_id, score = find_best_match(emb, index, product_ids)
            #log the match details for debugging 
            print(f"[DEBUG] Matching {det['class']} – Top product: {prod_id} (score: {round(score, 3)})")
            #store the matched produc info 
            products.append({
                "type": det["class"],#type of fashion items 
                "match_type": match_type,#match type 
                "matched_product_id": prod_id,#product ID from faiss match 
                "confidence": round(score, 3)#similarity score 
            })
    #step 7 - classify the vibes from the caption 
    vibes = classify_vibe(caption)
    #lop the classified vibes for debugging 
    print(f"[DEBUG] Vibes for caption '{caption}': {vibes}")
    #step 8 = combine everything into a result dictionary 
    result = {
        "video_id": video_id,
        "vibes": vibes,
        "products": products
    }
    #return the final result 
    return result
