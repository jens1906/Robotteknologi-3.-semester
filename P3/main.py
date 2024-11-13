import os
os.system('cls')

from Input import input as im
from Dehazing import dehaze as dh
from Palette_detection import LocateChecker as lc
from ColorCorrection import ColourCorrectMain as cc


import cv2 as cv
import numpy as np
from vmbpy import *
import matplotlib.pyplot as plt

def correct_image(image):
    return

def main():
    
    ## Get Image

    image = im.get_image()

    ## Dehazing

    dehazed_image = None
    dehazed_image = dh.dehaze(image)

    ## show both image and dehazed image on one plot
    dehazed_image = (dehazed_image * 255).astype(np.uint8)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[1].imshow(dehazed_image)
    axs[1].set_title('Dehazed Image')
    plt.show()

    ## Locate Color Checker
    if dehazed_image is None:
        dehazed_image = image
    print("non dehazed image: ")
    print(image[0])
    print("dehazed image: ")
    print(dehazed_image[0])
    checker, corner, pos = lc.LocateChecker(dehazed_image, cv.imread('P3\Palette_detection\Colour_checker_from_Vikki.png', cv.IMREAD_GRAYSCALE))
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[1].imshow(dehazed_image)
    axs[1].set_title('Dehazed Image')
    plt.show()
    

    ## Color Correction

    bad_pic = cv.imread('P3/ColorCorrection/U_Water_Sim_ColourUltimate.png')
    bad_pic = cv.cvtColor(bad_pic, cv.COLOR_BGR2RGB)

    ref_pal = cv.imread('P3/ColorCorrection/Color-Checker.jpg')
    ref_pal = cv.cvtColor(ref_pal, cv.COLOR_BGR2RGB)

    taken_pal = cv.imread('P3/ColorCorrection/U_Water_Sim_ColourUltimate.png')
    taken_pal = cv.cvtColor(taken_pal, cv.COLOR_BGR2RGB)

    corrected, cc_matrix = cc.colour_correct(bad_pic, ref_pal, taken_pal)

    print(cc_matrix)


    return

if __name__ == '__main__':
    main()
