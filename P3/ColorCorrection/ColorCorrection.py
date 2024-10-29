import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
global CCtesting
CCtesting = True
import os
os.system('cls')

print("Color Correction Testing:", CCtesting)
print("-----------------------------")

normboard = cv.imread('P3\ColorCorrection\Color-Checker.jpg')
normboard = cv.cvtColor(normboard, cv.COLOR_BGR2RGB)
#fuckboard = 
if CCtesting == True:
    plt.imshow(normboard) 
    #plt.show()

#find rgb value for each middle in each tile
rows = 4
cols = 6
pixwidth, pixheight, dim = normboard.shape
print("image size", pixwidth, pixheight)
tile_width = pixwidth // cols
tile_height = pixheight // rows

# Loop through each tile
rgb_matrix = []
for c in range(cols):
    for r in range(rows):
        # Calculate the middle pixel coordinates of the current tile, ensuring they are within bounds
        mid_x = min(c * tile_width + tile_width // 2, pixwidth - 1)
        mid_y = min(r * tile_height + tile_height // 2, pixheight - 1)
        #print("r, c, mid_x, mid_y", r, c, mid_x, mid_y)

        # Get the RGB values at the middle of the tile
        rgb_values = normboard[mid_x, mid_y]
        #print(f"Tile ({r+1}, {c+1}) - RGB: {rgb_values}")
        rgb_matrix.append(rgb_values)
print(rgb_matrix)
print(rgb_matrix[0])

 #plt.show()