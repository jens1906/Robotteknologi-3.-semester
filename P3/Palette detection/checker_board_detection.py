import cv2
import numpy as np
import time
import os

def display_checker(colour_checker):
        cv2.imshow("Colour Checker", colour_checker)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def display_box_checker(img,corners_img): # Draws a white box around the detected checkerboard
    img_with_box = cv2.polylines(img.copy(), [np.int32(corners_img)], True, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imshow("Detected Bounding Box", img_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scale_image(img, scale): # Scales the image by a factor of scale
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# Importing the images and template 
folder_path = "P3/Palette detection/Testpics"

dir_list = os.listdir(folder_path)
dir_list = sorted(dir_list, key=lambda x: int(x.split('.')[0]))
print(dir_list)

images = [cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE) for file in dir_list]
#images = [cv2.imread(P3/Palette detection/RotatedChecker/3D_transformed_img.png", cv2.IMREAD_GRAYSCALE)]
template = cv2.imread("P3/Palette detection/checker_board.PNG", cv2.IMREAD_GRAYSCALE)

# Start timer
tic = time.perf_counter()

# Initialize the ORB detector algorithm
orb = cv2.ORB_create()

def LocateChecker(img, template):
    img = scale_image(img, 0.5)
    
    # Detects features in the images in terms of keypoints and descriptors
    kp_img, desc_img = orb.detectAndCompute(img, None)
    kp_template, desc_template = orb.detectAndCompute(template, None)

    # Matches the features of the two images
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_template, desc_img)

    # Sort the matches based on distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)[:50]

    # Extract the matched keypoints
    points_template = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points_img = np.float32([kp_img[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    homography, mask = cv2.findHomography(points_template, points_img, cv2.RANSAC)

    # Get the dimensions of the template image
    h, w = template.shape[:2]

    # Define the corners of the template image and its corresponding corners in the target image
    corners_template = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_checker = cv2.perspectiveTransform(corners_template, homography)

    # Crop the detected colour checker from the img
    x_min, y_min = np.min(corners_checker[:, 0, :], axis=0).astype(int)
    x_max, y_max = np.max(corners_checker[:, 0, :], axis=0).astype(int)
    colour_checker = img[y_min:y_max, x_min:x_max]

    return colour_checker, corners_checker

for image in images:
    LocateChecker(image, template)

toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
