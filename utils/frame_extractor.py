import cv2
import os

def extract_frames(video_path, output_folder, seconds_interval=5):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30  # fallback if FPS can't be read

    interval = int(fps * seconds_interval)  # frames to skip = fps × seconds

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:03d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames every {seconds_interval} seconds.")
