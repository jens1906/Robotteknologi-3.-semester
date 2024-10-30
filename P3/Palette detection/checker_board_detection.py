import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os

folder_path = "P3/Palette detection/Testpics"
dir_list = os.listdir(folder_path)
print(dir_list)

images = [cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE) for file in dir_list]

template = cv2.imread("P3/Palette detection/checker_board.PNG", cv2.IMREAD_GRAYSCALE)

tic = time.perf_counter()

def LocateChecker(img, template):
    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create(1000)

    # Finds features in the images in terms of keypoints and descriptors
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

    # Define the corners of the template image
    corners_template = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Warp the corners of the template image to the target image
    corners_img = cv2.perspectiveTransform(corners_template.reshape(-1, 1, 2), homography)

    # Cropping img
    x_min, y_min = int(corners_img[0][0][0]), int(corners_img[0][0][1])
    x_max, y_max = int(corners_img[2][0][0]), int(corners_img[2][0][1])
    colour_checker = img[y_min:y_max, x_min:x_max]

    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

    def display_all():
        img_matches = cv2.drawMatches(template, kp_template, img, kp_img, matches[:5], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img_with_box = cv2.polylines(img, [np.int32(corners_img)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Matches", img_matches)
        cv2.imshow("Best Matched Object", img_with_box)
        cv2.imshow("Template Image", template)
        cv2.imshow("Colour Checker", colour_checker)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return colour_checker

template = LocateChecker(images[0], template)
template = LocateChecker(images[1]-images[0], template)
