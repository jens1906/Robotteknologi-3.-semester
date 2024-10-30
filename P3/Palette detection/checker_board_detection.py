import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os

folder_path = "C://Users//jens1//Documents//GitHub//Robotteknologi-3.-semester//P3//Palette detection//Testpics"
dir_list = os.listdir(folder_path)
print(dir_list)

images = []
for file in dir_list:
    image_path = os.path.join(folder_path, file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    images.append(image)

tic = time.perf_counter()

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')
#img = cv2.imread("P3/Palette detection/test_img.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("P3/Palette detection/checker_board.PNG", cv2.IMREAD_GRAYSCALE)

#img = cv2.imread("P3/Palette detection/test_img.jpg", cv2.IMREAD_GRAYSCALE)
#template = cv2.imread("P3/Palette detection/checker_board.PNG", cv2.IMREAD_GRAYSCALE)


def LocateChecker(img, template, imgNumber):
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
    LocChopped = [[int(corners_img[0][0][1]),
                  int(corners_img[2][0][1])]
                  ,[
                  int(corners_img[0][0][0]),
                  int(corners_img[2][0][0])]]

    print(f"{int(corners_img[0][0][1])},     {int(corners_img[2][0][1])},     {int(corners_img[0][0][0])},     {int(corners_img[2][0][0])}")
    # Display the images
    #cv2.imshow("Template Image", cv2.resize(template, (1080, 606), interpolation = cv2.INTER_LINEAR))
    #cv2.imshow("Target Image with Bounding Box", cv2.resize(img_with_box, (1080, 606), interpolation = cv2.INTER_LINEAR))
    cv2.imshow(f"Matches on: {imgNumber}", cv2.resize(img_matches, (1080, 606), interpolation = cv2.INTER_LINEAR))
    #cv2.imshow("Chopped", cv2.resize(chopped, (1080, 606), interpolation = cv2.INTER_LINEAR))
    
    return LocChopped

for i in range(len(images)):
    print(LocateChecker(images[i], template, i))
toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")


cv2.waitKey(0)
cv2.destroyAllWindows()