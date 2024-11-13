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
    plt.imshow(image)
    plt.show()

    ## Dehazing

    dehazed_image = None
    #dehazed_image = dh.dehaze(image)
    #plt.imshow(dehazed_image)
    #plt.show()

    ## Locate Color Checker
    if dehazed_image is None:
        dehazed_image = image

    checker = lc.LocateChecker(dehazed_image, 'P3/Palette_detection/ColorChecker.jpg')

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
