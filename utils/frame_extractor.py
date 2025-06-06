import cv2
import os

def extract_frames(video_path, output_folder, every_n_frames=1):
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    count, frame_id = 0, 0

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if count % every_n_frames == 0:
            filename = os.path.join(output_folder, f"frame_{frame_id:03d}.jpg")
            cv2.imwrite(filename, frame)
            frame_id += 1
        count += 1

    vidcap.release()
