import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
global CCtesting
CCtesting = False
import os
os.system('cls')

print("Color Correction Testing:", CCtesting)
print("-----------------------------")

normboard = cv.imread('P3\ColorCorrection\Color-Checker.jpg')
normboard = cv.cvtColor(normboard, cv.COLOR_BGR2RGB)
fuckboard = cv.imread('P3\ColorCorrection\Color-Checker-1.png')
fuckboard = cv.cvtColor(fuckboard, cv.COLOR_BGR2RGB)

if CCtesting == True:
    plt.imshow(normboard) 
    plt.show()

#find rgb value for each middle in each tile
def get_color_scheme(board):
    rows, cols = 4, 6
    imgheight, imgwidth, dim = board.shape
    #print("Image size:", imgwidth, imgheight)
    tile_width = imgwidth // cols
    tile_height = imgheight // rows


    rgb_list = []
    for r in range(rows):
        for c in range(cols):

            tilemidx, tilemidy = int(c * imgwidth / cols + (imgwidth / cols) / 2), int(r * imgheight / rows + (imgheight / rows) / 2)
            rgb_value = board[tilemidy, tilemidx]
            #print(r,c, tilemidx, tilemidy, rgb_value)
            rgb_list.append(rgb_value)
    return rgb_list

print(get_color_scheme(normboard)[19])
print(get_color_scheme(fuckboard)[19])
