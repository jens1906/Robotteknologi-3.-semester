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
#from Objective_testing import Objective_testing as ot

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
    Handles both individual images and lists of images.
    """
    # If a single list is passed, unpack it
    if len(images) == 1 and isinstance(images[0], list):
        images = images[0]
    
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
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    axs = axs.ravel()  # Flatten the array of axes for easy indexing
    
    for i, img in enumerate(images):
        if img.ndim == 2:  # Grayscale image
            axs[i].imshow(img, cmap='gray')
        else:  # Color image
            axs[i].imshow(img)
        #axs[i].axis('off')  # Hide axes
        # Use variable names as titles if they exist
        if i < len(image_names):
            axs[i].set_title(image_names[i], fontsize=25)
    
    # Hide any empty subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
<<<<<<< Updated upstream

    #remove most of the white space
    
=======
    #save plot in P3\Results\FullPlots
    timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
    plt.tight_layout()
    plt.savefig(f'P3/Results/FullPlots/FullPlot_{timestamp}.png')
>>>>>>> Stashed changes
    
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
    plt.savefig(f'P3/Results/FullPlots/FullPlot_{timestamp}.png', bbox_inches='tight')

    plt.show()


def main(cam=None, image_path=None, detailed=False):
    # Start time
    start_time = time.perf_counter()

    ## Get Image
    print("------Getting Image------")
    if cam is not None:
        try:
            frame = cam.get_frame() 
            bayer_image = frame.as_numpy_ndarray()
            image = cv.cvtColor(bayer_image, cv.COLOR_BAYER_BG2RGB)
        except Exception as e:
            raise Exception(f"Error capturing frame from camera: {e}")
    elif image_path is not None:
        image = cv.imread(image_path)
        if image is None:
            raise Exception("Error reading path")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    else:
        raise Exception("Camera or image path not initialized")
    
    #save image
    timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
    #cv.imwrite(f'P3/Results/OrgImages/image_{timestamp}.png', cv.cvtColor(image, cv.COLOR_BGR2RGB))
    
    #get image time
    image_time = time.perf_counter()


    ## Dehazing
    print("------Dehazing Image------")

    dehazed_image = dh.dehaze(image)

    dehaze_time = time.perf_counter()
    
    ## Locate Color dehazed_checker
    print("------Locating Color Checker------")

    template = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki_full.png', cv.IMREAD_GRAYSCALE)
    
    dehazed_checker, corner, pos = lc.LocateChecker(dehazed_image, template)



    locate_time = time.perf_counter()

    ## Color Correction
    print("------Color Correcting Image------")

    original_checker = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki.png')
    original_checker = cv.cvtColor(original_checker, cv.COLOR_BGR2RGB)

    try:
        corrected_image, cc_matrix, corrected_checker = cc.colour_correct(dehazed_image, original_checker, dehazed_checker)
        #corrected_image = cv.cvtColor(corrected_image, cv.COLOR_BGR2RGB)
    except:
        raise Exception("CC Failed")
    cc_time = time.perf_counter()

    print("------Image Processing Done------")

    ## End time
    end_time = time.perf_counter()
    if detailed:
        print(f"Image process took {end_time - start_time:.2f} seconds")
        print(f'Image loading took {image_time - start_time:.2f} seconds')
        print(f'Dehazing took {dehaze_time - image_time:.2f} seconds')
        print(f'Locating dehazed_checker took {locate_time - dehaze_time:.2f} seconds')
        print(f'Color correction took {cc_time - locate_time:.2f} seconds')

 
    ## Plot Images
    print("------Plotting Images------")
    if detailed:
        try:
            plot_images(image, dehazed_image, corrected_image, original_checker, dehazed_checker, corrected_checker, )
        except Exception as e:
            print("Error plotting images:", e)

    ## Objective Testing
    print("------Objective Testing------")
    #print("Image diff")
    #ot.ObjectiveTesting(corrected_image, image) #AG, MBE, PCQI
    #print("dehazed_checker diff")
    #apt.get_pal_diff(original_checker, dehazed_checker, corrected_checker) #Pal diff
    #print(cv.PSNR(original_checker, dehazed_checker))
    #print(cv.PSNR(original_checker, corrected_checker))


    print("------Finished------")
    return corrected_image

if __name__ == '__main__':
    cam = None
    image_path = None

    test_method = 'folder' # 'single', 'live', 'folder'

    if test_method == 'single':
<<<<<<< Updated upstream
        image_path = 'P3\Results\Data\Spinach30g\Beside_Camera_light10_exp49690.0_20242011_102755.png'
=======
        image_path = 'P3\Results\Data\Spinach20g\Beside_Camera_light5_20242011_094046.png'
>>>>>>> Stashed changes
        main(cam, image_path, True)

    elif test_method == 'live':
        with VmbSystem.get_instance() as vmb:
            while True:
                cams = vmb.get_all_cameras()
                if len(cams) == 0:
                    print('No cameras found')
                    exit(1)
                with cams[0] as cam:
                    try:
                        corrected = main(cam, None, True)
                        cv.imshow('Corrected Image', corrected)
                        if cv.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' press
                            break
                    except Exception as e:
                        print("Error during live processing:", e)
                        break
                cv.destroyAllWindows()
    
    elif test_method == 'folder':
        folder = 'P3\Results\Data\Spinach30g'
        os.makedirs(f'{folder}/Results', exist_ok=True)
        corrected_list = []
        for file in os.listdir(folder):
            if file.endswith('.png'):
                image_path = f'{folder}/{file}'
                print("!!!Processing: ", file)
                try:
                    corrected = main(cam, image_path)
                    corrected_list.append(corrected)
                    
                    timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
                    cv.imwrite(f'{folder}/Results/{file}_Result_{timestamp}_.png', cv.cvtColor(corrected, cv.COLOR_BGR2RGB))
                except Exception as e:
                    print("Failed", file, "Error:", e)
                    continue

        # Plot all corrected images
        plot_images(corrected_list)
    else:
        exit(1)
