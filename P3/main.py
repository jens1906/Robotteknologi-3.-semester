import os
import sys
os.system('cls')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Input import input as im
from Dehazing import dehaze as dh
from Palette_detection import LocateChecker as lc
from ColorCorrection import ColourCorrectMain as cc
from Objective_testing import APalTest as apt
from Objective_testing import Objective_testing as ot

import math
import time
import inspect
import cv2 as cv
import numpy as np
from vmbpy import *
from datetime import datetime
import matplotlib.pyplot as plt


def plot_images(*images):
    """
    Plots a given number of images in a grid layout using matplotlib.
    The image variable names from the caller's scope are used as titles.
    
    Parameters:
    - *images: a variable number of images (as arrays) to display.
    """

    #image, dehazed_image, corrected_image, ref_pal, checker, corrected_palette

    # Get the names of the variables from the caller's scope
    # Get the names of the variables from the caller's scope
    caller_locals = inspect.currentframe().f_back.f_locals
    image_names = []
    
    # Create a list of (name, image) pairs for each image passed into the function
    for image in images:
        for name, val in caller_locals.items():
            if isinstance(val, np.ndarray) and np.array_equal(val, image):
                image_names.append(name)
                break  # Once we find the name, stop searching
    
    # Number of images
    n = len(images)
    
    # Calculate grid size (rows and columns)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    # Create the plot
    fig, axs = plt.subplots(rows, cols)
    
    axs = axs.ravel()  # Flatten the array of axes for easy indexing
    
    for i, img in enumerate(images):
        axs[i].imshow(img)
        # Use variable names as titles if they exist
        if i < len(image_names):
            axs[i].set_title(image_names[i])

    # Hide any empty subplots
    for j in range(i + 1, rows * cols):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Start time
    start_time = time.perf_counter()

    ## Get Image
    print("------Getting Image------")
    
    if False:
        image = im.get_image()
    else:
        image = cv.imread('P3\Results\OrgImages\image_20241311_142217.png')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    #save image
    timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
    #cv.imwrite(f'P3/Results/OrgImages/image_{timestamp}.png', cv.cvtColor(image, cv.COLOR_BGR2RGB))
    #get image time
    image_time = time.perf_counter()


    ## Dehazing
    print("------Dehazing Image------")

    dehazed_image = dh.dehaze(image)

    if dehazed_image is None:
        dehazed_image = image

    dehaze_time = time.perf_counter()
    ## Locate Color Checker
    print("------Locating Color Checker------")

    template = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki.png', cv.IMREAD_GRAYSCALE)
    checker, corner, pos = lc.LocateChecker(image, template)    
    
    locate_time = time.perf_counter()

    ## Color Correction
    print("------Color Correcting Image------")

    ref_pal = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki.png')
    ref_pal = cv.cvtColor(ref_pal, cv.COLOR_BGR2RGB)

    try:
        corrected_image, cc_matrix, corrected_palette = cc.colour_correct(image, ref_pal, checker)
        #corrected_image = cv.cvtColor(corrected_image, cv.COLOR_BGR2RGB)
    except:
        raise Exception("CC Failed")
    cc_time = time.perf_counter()

    print("------Image Processing Done------")
    
    ## End time
    end_time = time.perf_counter()
    if False:
        print(f"Image process took {end_time - start_time:.2f} seconds")
        print(f'Image loading took {image_time - start_time:.2f} seconds')
        print(f'Dehazing took {dehaze_time - image_time:.2f} seconds')
        print(f'Locating checker took {locate_time - dehaze_time:.2f} seconds')
        print(f'Color correction took {cc_time - locate_time:.2f} seconds')

 
    ## Plot Images
    print("------Plotting Images------")
    plot_images(image, dehazed_image, corrected_image, ref_pal, checker, corrected_palette)


    ## Objective Testing
    print("------Objective Testing------")
    #print("Image diff")
    #ot.ObjectiveTesting(corrected_image, image) #AG, MBE, PCQI
    #print("Checker diff")
    #apt.get_pal_diff(ref_pal, checker, corrected_palette) #Pal diff
    #print(cv.PSNR(ref_pal, checker))
    #print(cv.PSNR(ref_pal, corrected_palette))


    print("------Finished------")
    return

if __name__ == '__main__':
    main()
