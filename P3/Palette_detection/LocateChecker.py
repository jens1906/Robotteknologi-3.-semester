import cv2
import numpy as np
import time
import os

test = False

if test == True:
    template = cv2.imread("P3/Palette_detection/Colour_checker_from_Vikki.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("P3/Results/OrgImages/image_20241311_142611.png")

    def display(colour_checker):
            cv2.imshow("Colour Checker", cv2.resize(colour_checker, None, fx=0.3, fy=0.4))    
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def display_box_checker(img,corners_img): # Draws a white box around the detected checkerboard
        img_with_box = cv2.polylines(img.copy(), [np.int32(corners_img)], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow("Detected Bounding Box", cv2.resize(img_with_box, None, fx=0.3, fy=0.4))            
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Initialize the ORB detector algorithm
def ORBLocateChecker(img, template, PreviousLocation=0, Adjustment=250, test=False):
    orb = cv2.ORB_create()
    if PreviousLocation != 0:
        # Adjust crop location by -250 for the top-left and +250 for the bottom-right
        X = [max(0, PreviousLocation[1][0] - Adjustment),
             min(img.shape[0], PreviousLocation[1][1] + Adjustment)]
        Y = [max(0, PreviousLocation[0][0] - Adjustment),
             min(img.shape[0], PreviousLocation[0][1] + Adjustment)]
        
        img = img[Y[0]: Y[1],   #y-axis (height)
                  X[0]: X[1]]   # x-axis (width)

    # Detects features in the images in terms of keypoints and descriptors
    kp_img, desc_img = orb.detectAndCompute(img, None)
    kp_template, desc_template = orb.detectAndCompute(template, None)

    # Matches the features of the two images
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_template, desc_img)
    img_matches = cv2.drawMatches(template, kp_template, img, kp_img, matches, None, 
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the resulting image with matches
    #cv2.imshow("Printed Matches pre", cv2.resize(img_matches, (640, 480)))
    #cv2.waitKey(0)

    # Sort the matches based on distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)[:100]

    display(cv2.drawMatches(template, kp_template, img, kp_img, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)) if test == True else None

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

    # Display the bounding box around the detected checkerboard
    display_box_checker(img, corners_checker) if test == True else None

    # Change the perspective of the detected colour checker to align it with the template
    warp_matrix = cv2.getPerspectiveTransform(corners_checker, corners_template)
    colour_checker = cv2.warpPerspective(img, warp_matrix, (w, h))

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
        FinalLocation = [[Y[0], Y[1]],
                         [X[0], X[1]]]

    # Display the cropped colour checker
    display(colour_checker) if test == True else None

    return colour_checker, corners_checker, FinalLocation


def AKAZELocateChecker(img, template, PreviousLocation=0, Adjustment=250, test=False):
    Akaze = cv2.AKAZE_create()  # Use SIFT for better results

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

    # Match features using KNN
    matcher = cv2.BFMatcher(cv2.NORM_L2)  # Use NORM_L2 for SIFT
    knn_matches = matcher.knnMatch(desc_template, desc_img, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print("Not enough good matches!")
        return None, None, None, None

    # Extract matched keypoints
    points_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_img = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(points_template, points_img, cv2.RANSAC)

    if homography is None: #or np.linalg.det(homography) < 1e-6
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

    colour_checker = colour_checker[170: 997,  # y-axis (height)
                                    520: 1705]   # x-axis (width)
    
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


def LocateChecker(img, template, PreviousLocation=0, Adjustment=100, test=False):
    #return ORBLocateChecker(img, template, PreviousLocation, Adjustment, test)
    return AKAZELocateChecker(img, template, PreviousLocation, Adjustment, test)

def LocateCheckerOriginal(img, template):
    colour_checker, corners, wrap_matrix, loc = LocateChecker(img, template)
    return colour_checker

###
### THIS IS TEST FUNCTIONS
###

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
    cap = cv2.VideoCapture("P3//Palette detection//Old program files//output_video.mp4")
    location = 0
    while True:
        success, img = cap.read()

        if img is None or img.shape == [0,0]:
            break
        else:
            #print(location)
            x,y,location = LocateChecker(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), template, location)
            cv2.imshow(f"Result:", cv2.resize(img, (968, 632)))
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


def DehazeTest():
    temp = cv2.imread("P3/Palette_detection/Colour_checker_from_Vikki_full.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("P3/Palette_detection/Dehazed_image.png")
    PIC,y,WARP,Location = LocateChecker(img, temp, 0, 250, True)
    cv2.imshow(f"Not cropped:", cv2.resize(PIC, (640, 480)))
    cv2.waitKey(0)

#DehazeTest()
#TestImageFolder()
#TestImageSmall()
#TestVideoFile()


# Previous code - ORB algorithm
"""

"""