from flask import Flask, request, jsonify
from main import run_pipeline
import os

app = Flask(__name__)
UPLOAD_FOLDER = "videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/process", methods=["POST"])
def process_video():
    file = request.files.get("video")
    caption = request.form.get("caption", "")
    
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video_id = filename.split(".")[0]
    file.save(video_path)

    try:
        result = run_pipeline(video_id, video_path, caption)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
