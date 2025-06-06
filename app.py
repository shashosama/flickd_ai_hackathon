import gradio as gr
import cv2
import os
from tempfile import NamedTemporaryFile

def extract_frames(video_path, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        _, im_encoded = cv2.imencode(".jpg", frame)
        frames.append(im_encoded.tobytes())
        count += 1
    cap.release()
    return frames

def analyze_video(video_file):
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    frame_bytes = extract_frames(tmp_path)
    results = []
    for i, fb in enumerate(frame_bytes):
        results.append((f"Frame {i}", fb, "mock_item", "casual" if i % 2 == 0 else "edgy"))

    return results

demo = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload a Video"),
    outputs=gr.Gallery(label="Detected Fashion Vibes").type("pil"),
    title=" Fashion Vibe Detector",
    description="Upload a video. We'll extract key frames and classify the fashion vibe (mock prediction)."
)

demo.launch()
