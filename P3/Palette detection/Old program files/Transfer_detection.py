import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os

TestingImages = False
Testing = False
template = cv2.imread("P3/Palette detection/checker_board.PNG", cv2.IMREAD_GRAYSCALE)
frameWidth = 640
frameHeight = 480
Adjustment = 100

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

tic = time.perf_counter()

def LocateChecker(img, template, imgNumber, PreviousLocation = 0):
    if PreviousLocation != 0:
        # Adjust crop location by -250 for the top-left and +250 for the bottom-right
        X = [max(0, PreviousLocation[1][0] - Adjustment),
             min(img.shape[0], PreviousLocation[1][1] + Adjustment)]
        Y = [max(0, PreviousLocation[0][0] - Adjustment),
             min(img.shape[0], PreviousLocation[0][1] + Adjustment)]
        
        img = img[Y[0]: Y[1],  # y-axis (height)
                  X[0]: X[1]]   # x-axis (width)

        cv2.imshow(f"Section:", img)

        if Testing:
            print(f"Previous location{imgNumber}: {PreviousLocation}")
            print(f"Max min stuff: {X[0]}, {Y[0]}, {X[1]}, {Y[1]}")
            print(f"Size of max min stuff: {(min(img.shape[0], PreviousLocation[0][1] + Adjustment)-max(0, PreviousLocation[0][0] - Adjustment))}, {(min(img.shape[0], PreviousLocation[1][1] + Adjustment)-max(0, PreviousLocation[1][0] - Adjustment))}")
            print(f"Current section size: {img.shape}")
            cv2.imshow(f"Section:", cv2.resize(img, (frameWidth, frameHeight)))
            cv2.imshow(f"Section:", img)
            cv2.waitKey(0)        
            print(img.shape)        
            cv2.waitKey(0)

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
    CurrentLocation = [[int(corners_img[0][0][1]),
                          int(corners_img[2][0][1])]
                        ,[int(corners_img[0][0][0]),
                          int(corners_img[2][0][0])]]
    #print(f"CurrentLocation: {CurrentLocation}")
    FinalLocation = CurrentLocation

    if PreviousLocation != 0:
        Y = [PreviousLocation[0][0]+(CurrentLocation[0][0]-Adjustment),#Y1
             PreviousLocation[0][1]+(CurrentLocation[0][0]-Adjustment)]#Y2
        X = [PreviousLocation[1][0]+(CurrentLocation[1][0]-Adjustment),#X1
             PreviousLocation[1][1]+(CurrentLocation[1][0]-Adjustment)]#X2
        #print(f"MiddleCalc{imgNumber}: {X[0]}, {X[1]}, {Y[0]}, {Y[1]}")        
        FinalLocation = [[Y[0],
                          Y[1]],
                         [X[0],
                          X[1]]]
        #print(f"Final Location{imgNumber}: {FinalLocation}")
        #print(f"Size{imgNumber}: {FinalLocation[0][1]-FinalLocation[0][0]}, {FinalLocation[1][1]-FinalLocation[1][0]}")


    chopped = img_with_box[int(corners_img[0][0][1]):int(corners_img[2][0][1]), int(corners_img[0][0][0]):int(corners_img[2][0][0])]

    if TestingImages:
        print(f"Current location{imgNumber}: {CurrentLocation}")
        print(f"{int(corners_img[0][0][1])},     {int(corners_img[2][0][1])},     {int(corners_img[0][0][0])},     {int(corners_img[2][0][0])}")
        # Display the images
        print("Images")
        cv2.imshow("Template Image", cv2.resize(template, (1080, 606), interpolation = cv2.INTER_LINEAR))
        cv2.imshow("Target Image with Bounding Box", cv2.resize(img_with_box, (1080, 606), interpolation = cv2.INTER_LINEAR))
        cv2.imshow("Matches", img_matches)
        cv2.imshow(f"Matches on: {imgNumber}", cv2.resize(img_matches, (1080, 606), interpolation = cv2.INTER_LINEAR))
        cv2.imshow(f"SmartChopped: {imgNumber}", cv2.resize(chopped, (1080, 606), interpolation = cv2.INTER_LINEAR))
    return FinalLocation

def TestImageFolder():
    folder_path = "P3//Palette detection//SmallMovement"
    location = 0
    dir_list = os.listdir(folder_path)
    dir_list = sorted(dir_list, key=lambda x: int(x.split('.')[0]))
    print(dir_list)

    images = []
    for file in dir_list:
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)
        
    for i in range(len(images)):
        location = LocateChecker(images[i], template, i, location)
        #print(location)
        cv2.waitKey(0)

def TestVideoFile():
    cap = cv2.VideoCapture("P3//Palette detection//output_video.mp4")
    i=0
    location = 0
    while True:
        success, img = cap.read()

        if img is None or img.shape == [0,0]:
            break
        else:
            #print(location)
            location = LocateChecker(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), template, i, location)
            i += 1
            cv2.imshow(f"Result:", cv2.resize(img, (frameWidth, frameHeight)))
            if cv2.waitKey(1) and 0xFF == ord('q'):
                break


TestVideoFile()
toc = time.perf_counter()
print(f"The program took {toc - tic:0.4f} seconds")


cv2.waitKey(0)
cv2.destroyAllWindows()