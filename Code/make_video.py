import cv2
import os

def make_video(images_folder, video_name, fps):
    # Adjust the sort key to handle 'render' prefix
    images = [img for img in os.listdir(images_folder) if img.startswith("render")]
    images.sort(key=lambda x: int(x.replace('render', '').split('.jpg')[0]))
    # Determine the width and height from the first image
    frame = cv2.imread(os.path.join(images_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(images_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Example usage
seq = 1  # Change scene number
images_folder = f'render_imgs/scene{seq}'
video_name = f'OutputVisualizationVideoSeq{seq}.mp4'
fps = 5  # Frames per second
make_video(images_folder, video_name, fps)
