import cv2 as cv
import numpy as np

# Define SRGB and IRGB as 3x8 matrices with example values
# Replace these with actual data for your specific case
SRGB = np.array([
    [115, 194, 98, 87, 133, 103, 214, 80],
    [82, 150, 122, 108, 128, 189, 126, 91],
    [68, 130, 157, 67, 177, 170, 44, 80]
])

IRGB = np.array([
    [120, 190, 95, 85, 130, 100, 210, 78],
    [85, 145, 120, 105, 125, 185, 120, 88],
    [70, 128, 155, 65, 175, 168, 40, 75]
])


def calculate_mcc(SRGB, IRGB):
    # Step 1: Compute SRGB * IRGB^T
    s_rgb_i_rgb_t = np.dot(SRGB, IRGB.T)
    
    # Step 2: Compute IRGB * IRGB^T
    i_rgb_i_rgb_t = np.dot(IRGB, IRGB.T)
    
    # Step 3: Take the inverse of IRGB * IRGB^T
    i_rgb_i_rgb_t_inv = np.linalg.inv(i_rgb_i_rgb_t)
    
    # Step 4: Compute M_CC
    M_CC = np.dot(s_rgb_i_rgb_t, i_rgb_i_rgb_t_inv)
    
    return M_CC

# Calculate the color correction matrix
M_CC = calculate_mcc(SRGB, IRGB)
print("Color Correction Matrix (M_CC):")
print(M_CC)