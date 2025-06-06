import gradio as gr
import os
import json
from tempfile import NamedTemporaryFile
from PIL import Image
from utils.frame_extractor import extract_frames
from main import run_pipeline

# Extract and return up to 6 preview frames
def extract_preview_frames(video_path, max_frames=6):
    extract_frames(video_path, "temp_frames", frame_rate=1)
    frames = []
    for i, file in enumerate(sorted(os.listdir("temp_frames"))):
        if i >= max_frames:
            break
        img = Image.open(os.path.join("temp_frames", file))
        frames.append(img)
    return frames

# Gradio handler
def analyze_video_gradio(video_file, caption):
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    result = run_pipeline(video_id=video_id, video_path=video_path, caption=caption)

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{video_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    preview_frames = extract_preview_frames(video_path)
    return preview_frames, f"Detected Vibes: {', '.join(result['vibes'])}"

# Launch UI
gr.Interface(
    fn=analyze_video_gradio,
    inputs=[
        gr.Video(label="Upload a Fashion Video"),
        gr.Textbox(label="Enter Caption / Hashtags")
    ],
    outputs=[
        gr.Gallery(label="Extracted Frames"),
        gr.Textbox(label="Vibes Detected")
    ],
    title=" Fashion Vibe Detector",
    description="Upload a video and caption. The AI extracts fashion items, finds matching products, classifies vibes, and saves results in `outputs/`."
).launch()
