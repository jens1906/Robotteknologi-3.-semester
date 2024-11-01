import cv2
import os

folder_path = "P3\Palette detection\SmallMovement"
output_video = "output_video.mp4"

# Get and sort the list of images in numerical order
images = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))
frame_rate = 2  # 2 frames per second (0.5 seconds per image)

# Load the first image to get dimensions
first_image = cv2.imread(os.path.join(folder_path, images[0]))
height, width, layers = first_image.shape

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Loop over each image, add it to the video
for image_name in images:
    image_path = os.path.join(folder_path, image_name)
    frame = cv2.imread(image_path)
    video.write(frame)  # Add frame to video

# Release the video writer
video.release()
print("Video created successfully!")
