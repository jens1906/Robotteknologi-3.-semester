import matlab.engine
import cv2
import numpy as np

# Load images using OpenCV
imageBefore = cv2.imread('Objective_testing/Objective_testing_before.jpg')
imageAfter = cv2.imread('Objective_testing/Objective_testing_after.jpg')

# Convert the OpenCV images to grayscale (MATLAB prefers grayscale images for NIQE and SSIM)
imageBefore_gray = cv2.cvtColor(imageBefore, cv2.COLOR_BGR2GRAY)
imageAfter_gray = cv2.cvtColor(imageAfter, cv2.COLOR_BGR2GRAY)

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Convert images from NumPy arrays to MATLAB arrays
imageBefore_matlab = matlab.double(imageBefore_gray.tolist())
imageAfter_matlab = matlab.double(imageAfter_gray.tolist())

# Display results
print("NIQE:", eng.niqe(imageBefore_matlab))
print("SSIM:", eng.ssim(imageBefore_matlab, imageAfter_matlab))
print("PIQE:", eng.piqe(imageBefore_matlab))

# Stop MATLAB engine
eng.quit()
