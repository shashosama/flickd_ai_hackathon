import os
import json                  
import cv2                 #for reading images as arrays  
from pathlib import Path   #for lisiting files in directories
from PIL import Image      #for image processing
import faiss               #for fast similarity search 
import numpy as np         #for math with vector 

#importing helper functions from your project 
from utils.frame_extractor import extract_frames             #extracts keyframes from video 
from models.yolo_detector import detect_fashion_items        #Detects fashion items in each frame
from models.clip_matcher import embed_image, find_best_match #Embeds + matches items 
from models.vibe_classifier import classify_vibe             #Classifies vibes from caption 

def run_pipeline(video_id, video_path, caption):     #the function runs your full pipeline on one video 
    #Extracts frames , detects fashion items , matches them to products, predicts vibes and return all results in JSON
    frame_dir = f"frames/{video_id}/"
    extract_frames(video_path, frame_dir)   #Extracts frames from video 

    # Load FAISS index + product IDs
    index = faiss.read_index("data/faiss_index.bin")
    with open("data/product_ids.txt") as f:
        product_ids = f.read().splitlines() # loads the saved product database (faiss) and the list of product IDs

    products = [] #this will store the result for each detected product
    for frame_file in Path(frame_dir).glob("*.jpg"):
        detections = detect_fashion_items(str(frame_file))
        img = cv2.imread(str(frame_file)) #loop through each frame and detect items using YOLO

        for det in detections:
            x, y, w, h = map(int, det["bbox"])
            x1, y1 = max(x - w // 2, 0), max(y - h // 2, 0)
            x2, y2 = x1 + w, y1 + h
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            #cropping the detected object using its bounding box 

            crop_img = Image.fromarray(crop)
            emb = embed_image(crop_img)
            #turns the cropped image into a vector (clip embedding)
            match_type, prod_id, score = find_best_match(emb, index, product_ids)
            #Use FAISS to find the most similar product in the catalog 
            if score < 0.75:
                continue

            products.append({
                "type": det["class"],
                "match_type": match_type,
                "matched_product_id": prod_id,
                "confidence": round(score, 3)
            })
            #save the product match result (type , match type , product ID and confidence score)

    vibes = classify_vibe(caption) # classify the vibe using the caption or hastags

    result = {
        "video_id": video_id,
        "vibes": vibes,
        "products": products
    }
    return result # return the final results as a dictionary 

if __name__ == "__main__":
    # Test
    result = run_pipeline(
        video_id="test01",
        video_path="data/test_video.mp4",
        caption="Feeling super Y2K and coquette with this new dress"
    )
    #runs the pipeline on a test video and saves the output to outputs/test01.json
    
    with open("outputs/test01.json", "w") as f:
        json.dump(result, f, indent=2)
