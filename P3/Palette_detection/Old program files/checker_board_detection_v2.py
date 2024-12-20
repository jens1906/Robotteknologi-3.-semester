import cv2
import numpy as np
import time
import os

Adjustment = 100
template = cv2.imread("P3/Palette detection/checker_board.PNG", cv2.IMREAD_GRAYSCALE)


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

# Start timer
tic = time.perf_counter()

# Initialize the ORB detector algorithm
orb = cv2.ORB_create()

def LocateChecker(img, template, PreviousLocation = 0):
    if PreviousLocation != 0:
        # Adjust crop location by -250 for the top-left and +250 for the bottom-right
        X = [max(0, PreviousLocation[1][0] - Adjustment),
             min(img.shape[0], PreviousLocation[1][1] + Adjustment)]
        Y = [max(0, PreviousLocation[0][0] - Adjustment),
             min(img.shape[0], PreviousLocation[0][1] + Adjustment)]
        
        img = img[Y[0]: Y[1],  # y-axis (height)
                  X[0]: X[1]]   # x-axis (width)

        cv2.imshow(f"Section:", img)


    #img = scale_image(img, 0.5)
    
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

    # Chopped image
    CurrentLocation = [[int(corners_checker[0][0][1]),
                          int(corners_checker[2][0][1])]
                        ,[int(corners_checker[0][0][0]),
                          int(corners_checker[2][0][0])]]
    FinalLocation = CurrentLocation

    if PreviousLocation != 0:
        Y = [PreviousLocation[0][0]+(CurrentLocation[0][0]-Adjustment),#Y1
             PreviousLocation[0][1]+(CurrentLocation[0][0]-Adjustment)]#Y2
        X = [PreviousLocation[1][0]+(CurrentLocation[1][0]-Adjustment),#X1
             PreviousLocation[1][1]+(CurrentLocation[1][0]-Adjustment)]#X2
        FinalLocation = [[Y[0],
                          Y[1]],
                         [X[0],
                          X[1]]]

    #cv2.waitKey(0)
    return colour_checker, corners_checker, FinalLocation






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
        x,y,location = LocateChecker(images[i], template, location)
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
            x,y,location = LocateChecker(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), template, location)
            i += 1
            cv2.imshow(f"Result:", cv2.resize(img, (640, 480)))
            if cv2.waitKey(1) and 0xFF == ord('q'):
                break


def TestImageSmall():
    # Importing the images and template 
    folder_path = "P3/Palette detection/Testpics"

    dir_list = os.listdir(folder_path)
    dir_list = sorted(dir_list, key=lambda x: int(x.split('.')[0]))
    print(dir_list)

    images = [cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE) for file in dir_list]
    #images = [cv2.imread(P3/Palette detection/RotatedChecker/3D_transformed_img.png", cv2.IMREAD_GRAYSCALE)]
    Location = 0
    for image in images:
        x,y,Location = LocateChecker(image, template, Location)




#TestImageFolder()
#TestImageSmall()
TestVideoFile()

toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")