import os
import cv2

def extract_frames(video_path, output_path, skip_interval=10):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    
    # Get the frame count of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the step size to extract every 10th frame
    step_size = skip_interval
    
    frame_count = 0  # Start frame numbering from 0
    extracted_frame_count = 1  # Start extracted frame numbering from 1
    
    while True:
        # Set the frame position to the next frame to extract
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        # Read the next frame
        success, frame = cap.read()
        
        # Check if we have reached the end of the video
        if not success:
            break
        
        # Save the frame with sequential numbering
        frame_name = f"frame{extracted_frame_count}.jpg"
        frame_path = os.path.join(output_path, frame_name)
        cv2.imwrite(frame_path, frame)
        
        # Increment frame count by step size
        frame_count += step_size
        extracted_frame_count += 1
        
        # Check if we have reached the end of the video
        if frame_count >= total_frames:
            break
    
    # Release the VideoCapture object
    cap.release()
    print("Frames extracted successfully.")


data_path = 'C:/Users/msult/Downloads/P3Data/P3Data/Sequences'
base_frames_path = 'C:/Users/msult/Downloads/P3Data/P3Data'

frames_directory = os.path.join(base_frames_path, "Frames")
if not os.path.exists(frames_directory):
    os.makedirs(frames_directory)

for i in range(1, 14):
    scene_dir = os.path.join(frames_directory, f'scene{i}')
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)
    
    video_path = "C:\\Users\\msult\\Downloads\\P3Data\\P3Data\\Sequences\\scene1\\Undist\\2023-02-14_11-04-07-front_undistort.mp4"
    video_base_path = os.path.join(data_path, f'scene{i}', 'Undist')
    video_list = os.listdir(video_base_path)
    for name in video_list:
        if name[-19:] == 'front_undistort.mp4':
            video_path = os.path.join(video_base_path, name)

    extract_frames(video_path, scene_dir, skip_interval=5)