import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

global CCtesting
CCtesting = True
os.system('cls')

if CCtesting:
    print("Color Correction Testing:", CCtesting)
    print("---------------Import Test Boards--------------")
    normboard = cv.imread('P3/ColorCorrection/Color-Checker.jpg')  # Use '/' for paths
    normboard = cv.cvtColor(normboard, cv.COLOR_BGR2RGB)
    fuckboard = cv.imread('P3/ColorCorrection/Color-Checker-1.png')  # Use '/' for paths
    fuckboard = cv.cvtColor(fuckboard, cv.COLOR_BGR2RGB)

    f, axarr = plt.subplots(1, 2)  # Creates a 1x2 grid
    axarr[0].imshow(normboard)
    axarr[0].set_title("Original Color Checker")
    axarr[1].imshow(fuckboard)
    axarr[1].set_title("Modified Color Checker")
    #plt.show()

# Find RGB value for each middle in each tile
def get_color_scheme(board):
    rows, cols = 4, 6
    imgheight, imgwidth, dim = board.shape
    tile_width = imgwidth // cols
    tile_height = imgheight // rows

    rgb_list = []
    for r in range(rows):
        for c in range(cols):
            tilemidx = int(c * imgwidth / cols + (imgwidth / cols) / 2)
            tilemidy = int(r * imgheight / rows + (imgheight / rows) / 2)
            rgb_value = board[tilemidy, tilemidx]
            rgb_list.append(rgb_value)
    
    # Return a NumPy array instead of a list
    return np.array(rgb_list)

if CCtesting:
    print("---------------Test get_color_scheme--------------")
    print("normboard colors:")
    print(get_color_scheme(normboard))
    print("fuckboard colors:")
    print(get_color_scheme(fuckboard))

def calculate_mcc(SRGB, IRGB):
    # Step 1: Compute SRGB * IRGB^T
    s_rgb_i_rgb_t = np.dot(SRGB.T, IRGB)
    
    # Step 2: Compute IRGB * IRGB^T
    i_rgb_i_rgb_t = np.dot(IRGB.T, IRGB)
    
    # Step 3: Take the inverse of IRGB * IRGB^T
    i_rgb_i_rgb_t_inv = np.linalg.inv(i_rgb_i_rgb_t)
    
    # Step 4: Compute M_CC
    M_CC = np.dot(s_rgb_i_rgb_t, i_rgb_i_rgb_t_inv)
    
    return M_CC

if CCtesting == True:
    print("---------------Test calculate_mcc--------------")
    print(calculate_mcc(get_color_scheme(normboard), get_color_scheme(fuckboard)))
