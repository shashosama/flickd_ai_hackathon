import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(fps // frame_rate) if fps > 0 else 1

    count, frame_id = 0, 0
    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if count % interval == 0:
            filename = os.path.join(output_folder, f"frame_{frame_id:03d}.jpg")
