import gradio as gr
import cv2
import os
from tempfile import NamedTemporaryFile
from PIL import Image
import numpy as np
import io

def extract_frames(video_path, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
        frames.append(Image.fromarray(frame))
        count += 1
    cap.release()
    return frames

def analyze_video(video_file):
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    frame_images = extract_frames(tmp_path)
    return frame_images  # Returning list of PIL images

gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload a Video"),
    outputs=gr.Gallery(label="Detected Fashion Vibes", show_label=True, columns=2),
    title="👗 Fashion Vibe Detector",
    description="Upload a video. We extract key frames and show a mock prediction of fashion vibes."
).launch()

