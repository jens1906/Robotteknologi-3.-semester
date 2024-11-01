import cv2 as cv
import numpy as np


# Define SRGB and IRGB as 1x8 arrays, where each element is an RGB tuple
# Each tuple represents one color in the format (R, G, B)
SRGB = np.array([
    (115, 82, 68),
    (194, 150, 130),
    (98, 122, 157),
    (87, 108, 67),
    (133, 128, 177),
    (103, 189, 170),
    (214, 126, 44),
    (80, 91, 80)
])

IRGB = np.array([
    (120, 85, 70),
    (190, 145, 128),
    (95, 120, 155),
    (85, 105, 65),
    (130, 125, 175),
    (100, 185, 168),
    (210, 120, 40),
    (78, 88, 75)
])


def calculate_mcc(SRGB, IRGB):
    # Convert SRGB and IRGB to 8x3 matrices for matrix calculations
    SRGB_matrix = np.vstack(SRGB)
    IRGB_matrix = np.vstack(IRGB)
    
    # Step 1: Compute SRGB^T * IRGB
    s_rgb_t_i_rgb = np.dot(SRGB_matrix.T, IRGB_matrix)
    
    # Step 2: Compute IRGB^T * IRGB
    i_rgb_t_i_rgb = np.dot(IRGB_matrix.T, IRGB_matrix)
    
    # Step 3: Take the inverse of IRGB^T * IRGB
    i_rgb_t_i_rgb_inv = np.linalg.inv(i_rgb_t_i_rgb)
    
    # Step 4: Compute M_CC
    M_CC = np.dot(s_rgb_t_i_rgb, i_rgb_t_i_rgb_inv)
    
    return M_CC

# Calculate the color correction matrix
M_CC = calculate_mcc(SRGB, IRGB)
print("Color Correction Matrix (M_CC):")
print(M_CC)
