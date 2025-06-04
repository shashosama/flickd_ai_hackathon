from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

from utils.frame_extractor import extract_frames
from models.yolo_detector import detect_fashion_items
from models.clip_matcher import embed_image, find_best_match
from models.vibe_classifier import classify_vibe

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_video():
    video = request.files.get('file')
    if not video:
        return jsonify({"error": "No video file uploaded"}), 400

    filename = secure_filename(video.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video.save(video_path)

    try:
        # Step 1: Extract frames
        frames = extract_frames(video_path)

        results = []
        for frame in frames:
            items = detect_fashion_items(frame)
            for item in items:
                embedding = embed_image(item['cropped_image'])
                matched_product = find_best_match(embedding)
                vibe = classify_vibe(item['cropped_image'])

                results.append({
                    "label": item['label'],
                    "bbox": item['bbox'],
                    "matched_product": matched_product,
                    "vibe": vibe
                })

        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
