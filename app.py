import gradio as gr
import os
import json
from tempfile import NamedTemporaryFile
from PIL import Image
#improt custom utility functions 
from utils.frame_extractor import extract_frames
from main import run_pipeline
#define a helper funtion to extract  and return preview frames from a video 
# Extract and return up to 6 preview frames
def extract_preview_frames(video_path, max_frames=6):
    #use the  custom function to extract frames into the 'temp_frames' directory 
    extract_frames(video_path, "temp_frames", seconds_interval=1)
    #store extracted frames here 
    frames = []
    #loop through the sorted frame filenames 
    for i, file in enumerate(sorted(os.listdir("temp_frames"))):
        if i >= max_frames:
            break#stop after collecting max_frames 
        #open and append the image to the list
        img = Image.open(os.path.join("temp_frames", file))
        frames.append(img)
    return frames

# Gradio handler
#define the gradio interface handler that runs when a user submits input 
def analyze_video_gradio(video_file, caption):
    # Accept both uploaded binary and file path
    #handle both direct file paths and uploaded binary files 
    if isinstance(video_file, str):
        video_path = video_file
    else:
        #save the uploaded binary video file to a temproary location 
        with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name
#generate a unique video ID from the filename 
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    #run the main analysis pipeline : detection , matching , vibe classification 
    result = run_pipeline(video_id=video_id, video_path=video_path, caption=caption)
    #save the result dictionary as json file in the outputs folder 

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{video_id}.json", "w") as f:
        json.dump(result, f, indent=2)
#extract preview frames from the video to show in the UI 
    preview_frames = extract_preview_frames(video_path)
    #return preview frames and a string of detected vides 
    return preview_frames, f"Detected Vibes: {', '.join(result['vibes'])}"

# Launch UI
gr.Interface(
    fn=analyze_video_gradio, #function to call when user submits input 
    inputs=[
        gr.Video(label="Upload a Fashion Video"),#video upload input
        gr.Textbox(label="Enter Caption / Hashtags")#caption inout for vibes 
    ],
    outputs=[
        gr.Gallery(label="Extracted Frames"),#output of gallery of extracted video frames 
        gr.Textbox(label="Vibes Detected")#output = detected fashion vibes 
    ],
    title=" Fashion Vibe Detector",#title of the web app 
    description="Upload a video and caption. The AI extracts fashion items, finds matching products, classifies vibes, and saves results in `outputs/`."
).launch()
