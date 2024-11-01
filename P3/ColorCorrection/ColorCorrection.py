import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
global CCtesting
CCtesting = True

print("Color Correction Testing:", CCtesting)

if CCtesting == True:
    normboard = cv.imread('P3\ColorCorrection\Color-Checker.jpg')
    #fuckboard = 
    plt.imshow(normboard) 
    plt.show()

#print hello world
print("he")