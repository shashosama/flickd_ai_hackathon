from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from main import run_pipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_frames(video_path, output_folder, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_paths = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f'frame_{frame_count}.jpg'
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        saved_paths.append(frame_path)
        frame_count += 1

    cap.release()
    return saved_paths

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Upload a video to analyze."})

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"status": "failed", "message": "No file part in request."}), 400

    video = request.files['file']
    if video.filename == '':
        return jsonify({"status": "failed", "message": "No selected file."}), 400

    try:
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        frame_paths = extract_frames(video_path, app.config['UPLOAD_FOLDER'])

        results = [
            {
                "frame": os.path.basename(p),
                "fashion_item": f"mock_item_{i}",
                "vibe": "casual" if i % 2 == 0 else "edgy"
            }
            for i, p in enumerate(frame_paths)
        ]

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
