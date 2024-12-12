import cv2
import numpy as np
import time
import os

def LocateChecker(img, template, PreviousLocation=0, Adjustment=250, test=False, original = False):
    Akaze = cv2.AKAZE_create()  # Use SIFT for better results
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if PreviousLocation != 0:
        # Adjust crop location
        X = [max(0, PreviousLocation[1][0] - Adjustment),
             min(img.shape[1], PreviousLocation[1][1] + Adjustment)]
        Y = [max(0, PreviousLocation[0][0] - Adjustment),
             min(img.shape[0], PreviousLocation[0][1] + Adjustment)]
        img = img[Y[0]: Y[1],   #y-axis (height)
                  X[0]: X[1]]   #x-axis (width)

    # Detect features
    kp_img, desc_img = Akaze.detectAndCompute(img, None)
    kp_template, desc_template = Akaze.detectAndCompute(template, None)

    # Check for None descriptors
    if desc_img is None or desc_template is None and original:
        print("Descriptors not found! Check input images.")
        return None, None, None, None

    # Match features using KNN
    matcher = cv2.BFMatcher(cv2.NORM_L2)  # Use NORM_L2 for SIFT
    knn_matches = matcher.knnMatch(desc_template, desc_img, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.80 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4 and original:
        print("Not enough good matches!")
        return None, None, None, None

    # Extract matched keypoints
    points_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_img = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(points_template, points_img, cv2.RANSAC)

    if homography is None and original: #or np.linalg.det(homography) < 1e-6
        print("Invalid homography matrix!")
        return None, None, None, None

    # Transform corners
    h, w = template.shape[:2]
    corners_template = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_checker = cv2.perspectiveTransform(corners_template, homography)
    img_with_box = cv2.polylines(img.copy(), [np.int32(corners_checker)], True, (0, 0, 0), 3, cv2.LINE_AA)
    #cv2.imwrite("AKAZE_Box.png", img_with_box)
    #cv2.imshow("Detected Bounding Box", cv2.resize(img_with_box, (640, 480)))
    #cv2.waitKey(0)

    if corners_checker is None or len(corners_checker) != 4:
        print("Invalid corners detected!")
        return None, None, None, None

    # Warp perspective
    warp_matrix = cv2.getPerspectiveTransform(corners_checker, corners_template)
    colour_checker = cv2.warpPerspective(img, warp_matrix, (w, h))
    #cv2.imwrite("AKAZE_Warped.png", colour_checker)

    #colour_checker = colour_checker[170: 997,  # y-axis (height)
    #                                520: 1705]   # x-axis (width)
    colour_checker = colour_checker[47: 270,  # y-axis (height)
                                    139: 459]   # x-axis (width)
    
    # Debugging
    if test:
        img_matches = cv2.drawMatches(template, kp_template, img, kp_img, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Good Matches",  cv2.resize(img_matches, (640, 480)))
        #cv2.imwrite("AKAZE_Matches.png", img_matches)
        cv2.imshow("Warped Colour Checker", cv2.resize(colour_checker, (640, 480)))
        #cv2.imwrite("AKAZE_Warped.png", colour_checker)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return colour_checker, corners_checker, warp_matrix, [[int(corners_checker[0][0][1]), int(corners_checker[2][0][1])],
                                                        [int(corners_checker[0][0][0]), int(corners_checker[2][0][0])]]


def LocateCheckerOriginal(img, template, warp_matrix):
    # Transform corners
    h, w = template.shape[:2]
   
    # Warp perspective
    colour_checker = cv2.warpPerspective(img, warp_matrix, (w, h))
    #colour_checker = colour_checker[170: 997,  # y-axis (height)
    #                                520: 1705]   # x-axis (width)
    colour_checker = colour_checker[47: 270,  # y-axis (height)
                                    139: 459]   # x-axis (width)    
    
    return colour_checker

