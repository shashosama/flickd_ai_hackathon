from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_frames(video_path, output_folder, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_paths = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
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
    video = request.files.get('file')
    if video:
        filename = secure_filename(video.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(video_path)

        frame_paths = extract_frames(video_path, UPLOAD_FOLDER)

        # Simulate fashion tag + vibe prediction
        results = [
            {
                "frame": os.path.basename(p),
                "fashion_item": "mock_item_" + str(i),
                "vibe": "casual" if i % 2 == 0 else "edgy"
            }
            for i, p in enumerate(frame_paths)
        ]

        return jsonify({"status": "success", "results": results})

    return jsonify({"status": "failed", "message": "No file uploaded."})

if __name__ == '__main__':
    app.run(debug=True)
