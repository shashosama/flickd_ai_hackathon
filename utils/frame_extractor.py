import cv2
import os
from utils.frame_extractor import extract_frames
# importing these to process video and hep with folder creation and file path handling


def extract_frames(video_path, output_folder, frame_rate=1):
    #defines extract_frames that : takes in the path to the video, the output folder to save extracted frames and extracts 1 frame per sec 

    os.makedirs(output_folder, exist_ok=True)# creates an output folder if it doesnt exist 
    vidcap = cv2.VideoCapture(video_path)# opens the vedio file for reading 
    fps = vidcap.get(cv2.CAP_PROP_FPS)#gets the videos frame rate 
    interval = int(fps // frame_rate) if fps > 0 else 1
    #calculates how often to grab a frame . if fps=30 and framerate =1 , interval =30. So one frame every 30 frames = 1 perseond 

    count, frame_id = 0, 0 
    #count= total frames read so far , frame_id = number of saved frames 
    while True:
        success, frame = vidcap.read()
        #reads each frame of the video 
        #if no frame is left , break the loop
        if not success:
            break
        if count % interval == 0:
            #save every interval-th frame 
            filename = os.path.join(output_folder, f"frame_{frame_id:03d}.jpg")
            #builds the output file name like : frame_000 , or frame_001
            #writes(saves) the current frames as an image file 
            cv2.imwrite(filename, frame)
            frame_id += 1
            #increments the frame save counter 
            count += 1
            #increment the total frame read counter 
            vidcap.release()
            #release the video file 
