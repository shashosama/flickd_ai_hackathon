from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from utils.frame_extractor import extract_frames
from models.yolo_detector import detect_fashion_items
from models.clip_matcher import embed_image, find_best_match
from models.vibe_classifier import classify_vibe


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video = request.files['file']
    if video:
        filename = secure_filename(video.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(path)
        # run your video + CLIP + YOLO pipeline here
        return jsonify({"status": "success", "file": filename})
    return jsonify({"status": "failed"})

if __name__ == '__main__':
    app.run(debug=True)
return render_template('index.html')

