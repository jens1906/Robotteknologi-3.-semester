import os
os.system('cls')

from Input import input as im
from Dehazing import dehaze as dh
from Palette_detection import LocateChecker as lc
from ColorCorrection import ColourCorrectMain as cc


import cv2 as cv
import numpy as np
from vmbpy import *
from datetime import datetime
import matplotlib.pyplot as plt

def correct_image(image):
    return

def main():
    
    ## Get Image
    print("------Getting Image------")

    image = im.get_image()

    ## Dehazing
    print("------Dehazing Image------")

    dehazed_image = None
    dehazed_image = dh.dehaze(image)

    ## show both image and dehazed image on one plot
    """
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[1].imshow(dehazed_image)
    axs[1].set_title('Dehazed Image')
    plt.show()
    """
    if dehazed_image is None:
        dehazed_image = image

    ## Locate Color Checker
    print("------Locating Color Checker------")

    template = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki.png', cv.IMREAD_GRAYSCALE)
    checker, corner, pos = lc.LocateChecker(image, template)

    #save image in P3/Results/OrgImages as png witht the name image[date-time].png
    timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
    cv.imwrite(f'P3/Results/OrgImages/image_{timestamp}.png', cv.cvtColor(image, cv.COLOR_BGR2RGB))
    
    ## Color Correction
    print("------Color Correcting Image------")

    ref_pal = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki.png')
    ref_pal = cv.cvtColor(ref_pal, cv.COLOR_BGR2RGB)

    corrected, cc_matrix = cc.colour_correct(image, ref_pal, checker)
    corrected = cv.cvtColor(corrected, cv.COLOR_BGR2RGB)

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original Image')
    axs[0, 1].imshow(dehazed_image)
    axs[0, 1].set_title('Dehazed Image')
    axs[0, 2].imshow(cv.cvtColor(corrected, cv.COLOR_BGR2RGB))
    axs[0, 2].set_title('CC Image')
    axs[1, 0].imshow(ref_pal)
    axs[1, 0].set_title('Reference Palette')
    axs[1, 1].imshow(checker)
    axs[1, 1].set_title('Checker Image')
    axs[1, 2].imshow(corner)
    plt.show()

    print(cc_matrix)



    return

if __name__ == '__main__':
    main()
