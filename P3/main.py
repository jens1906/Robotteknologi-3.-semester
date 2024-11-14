import os
os.system('cls')
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Input import input as im
from Dehazing import dehaze as dh
from Palette_detection import LocateChecker as lc
from ColorCorrection import ColourCorrectMain as cc
from Objective_testing import APalTest as apt



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

    if False:
        image = im.get_image()
    else:
        image = cv.imread('P3/Results/OrgImages/image_20241411_085252.png')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    #save image
    timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
    #cv.imwrite(f'P3/Results/OrgImages/image_{timestamp}.png', cv.cvtColor(image, cv.COLOR_BGR2RGB))

    ## Dehazing
    print("------Dehazing Image------")

    dehazed_image = dh.dehaze(image)

    if dehazed_image is None:
        dehazed_image = image

    ## Locate Color Checker
    print("------Locating Color Checker------")

    template = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki.png', cv.IMREAD_GRAYSCALE)
    checker, corner, pos = lc.LocateChecker(image, template)    
    
    ## Color Correction
    print("------Color Correcting Image------")

    ref_pal = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki.png')
    ref_pal = cv.cvtColor(ref_pal, cv.COLOR_BGR2RGB)

    try:
        corrected_image, cc_matrix, corrected_palette = cc.colour_correct(image, ref_pal, checker)
        corrected_image = cv.cvtColor(corrected_image, cv.COLOR_BGR2RGB)
    except:
        raise Exception("CC Failed")
    
    ## Get Palette Difference
    apt.get_pal_diff(ref_pal, checker, corrected_palette)

    # Plot images
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original Image')
    axs[0, 1].imshow(dehazed_image)
    axs[0, 1].set_title('Dehazed Image')
    axs[0, 2].imshow(cv.cvtColor(corrected_image, cv.COLOR_BGR2RGB))
    axs[0, 2].set_title('CC Image')
    axs[1, 0].imshow(ref_pal)
    axs[1, 0].set_title('Reference Palette')
    axs[1, 1].imshow(checker)
    axs[1, 1].set_title('Found Palette')
    axs[1, 2].imshow(corrected_palette)
    axs[1, 2].set_title('CC Palette')
    plt.savefig(f'P3/Results/FullPlots/FullPlot_{timestamp}.png')
    plt.show()
    

    print("------Finished------")
    return

if __name__ == '__main__':
    main()
