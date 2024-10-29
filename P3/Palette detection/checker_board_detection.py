import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

tic = time.perf_counter()

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

img = cv2.imread("P3/Palette detection/test_img.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("P3/Palette detection/checker_board.PNG", cv2.IMREAD_GRAYSCALE)


# Initialize the ORB detector algorithm
orb = cv2.ORB_create()

# Finds features in the images in terms of keypoints and descriptors
keypoints_img, descriptors_img = orb.detectAndCompute(img, None)
keypoints_template, descriptors_template = orb.detectAndCompute(template, None)

# Matches the features of the two images
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors_template, descriptors_img)

# Sort the matches based on distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top matches
img_matches = cv2.drawMatches(template, keypoints_template, img, keypoints_img, matches[:5], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Extract the matched keypoints
points_template = np.zeros((len(matches), 2), dtype=np.float32)
points_img = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points_template[i, :] = keypoints_template[match.queryIdx].pt
    points_img[i, :] = keypoints_img[match.trainIdx].pt

# Find the homography matrix
homography, mask = cv2.findHomography(points_template, points_img, cv2.RANSAC)

# Get the dimensions of the template image
h, w = template.shape[:2]

# Define the corners of the template image
corners_template = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)

# Warp the corners of the template image to the target image
corners_img = cv2.perspectiveTransform(corners_template, homography)

# Draw the bounding box around the detected square
img_with_box = cv2.polylines(img, [np.int32(corners_img)], True, (0, 255, 0), 3, cv2.LINE_AA)

# Chopped image
chopped = img_with_box[int(corners_img[0][0][1]):int(corners_img[2][0][1]), int(corners_img[0][0][0]):int(corners_img[2][0][0])]

# Display the images
cv2.imshow("Template Image", template)
cv2.imshow("Target Image with Bounding Box", img_with_box)
cv2.imshow("Matches", img_matches)
cv2.imshow("Chopped", chopped)

toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")


cv2.waitKey(0)
cv2.destroyAllWindows()