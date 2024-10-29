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
#fuckboard = 
if CCtesting == True:
    
    plt.imshow(normboard) 
    #plt.show()

#find rgb value for each middle in each tile
rows = 4
cols = 6
pixwidth, pixheight, dim = normboard.shape
print(pixwidth, pixheight)
tile_width = pixwidth // cols
tile_height = pixheight // rows

for c in range(cols):
    for r in range(rows):
        mid_x = c * tile_width + tile_width // 2
        mid_y = r * tile_height + tile_height // 2

        print(mid_x, mid_y)
        rgb_values = normboard[mid_y, mid_x]
        print(f"Tile ({r+1}, {c+1}) - RGB: {rgb_values}")
        print("---------")

 

 #plt.show()