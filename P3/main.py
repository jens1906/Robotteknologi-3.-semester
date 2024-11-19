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
            axs[i].set_title(image_names[i], fontsize=8)
    
    # Hide any empty subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
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

    if dehazed_image is None:
        dehazed_image = image

    dehaze_time = time.perf_counter()
    
    ## Locate Color Checker
    print("------Locating Color Checker------")

    template = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki_full.png', cv.IMREAD_GRAYSCALE)

    checker, corner, pos = lc.LocateChecker(dehazed_image, template)    
    
    locate_time = time.perf_counter()

    ## Color Correction
    print("------Color Correcting Image------")

    ref_pal = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki.png')
    ref_pal = cv.cvtColor(ref_pal, cv.COLOR_BGR2RGB)

    try:
        corrected_image, cc_matrix, corrected_palette = cc.colour_correct(dehazed_image, ref_pal, checker)
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
        print(f'Locating checker took {locate_time - dehaze_time:.2f} seconds')
        print(f'Color correction took {cc_time - locate_time:.2f} seconds')

 
    ## Plot Images
    print("------Plotting Images------")
    if detailed:
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
    return corrected_image

if __name__ == '__main__':
    cam = None
    image_path = None


    test_method = 'single'


    if test_method == 'single':
        image_path = 'P3/Results/Data/16th_Milk/Beside_Camera_20241611_120010.png'
        main(cam, image_path, True)

    if test_method == 'live':
        from vmbpy import Vimba  # Import the Vimba API context manager
    
        with Vimba() as vimba:  # Start the Vimba API
            with im.initialize_camera(vimba) as cam:  # Ensure the camera is initialized correctly
                while True:
                    try:
                        corrected = main(cam, image_path, True)
                        cv.imshow('Corrected Image', corrected)
                        if cv.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' press
                            break
                    except Exception as e:
                        print("Error during live processing:", e)
                        break
                cv.destroyAllWindows()
    
    elif test_method == 'folder':
        folder = 'P3/Results/Data/32th_Milk'
        corrected_list = []
        for file in os.listdir(folder):
            if file.endswith('.png'):
                image_path = f'{folder}/{file}'
                print("!!!!!!!!!!!Processing: ", file)
                try:
                    corrected = main(cam, image_path)
                    corrected_list.append(corrected)
                except Exception as e:
                    print("Failed", file, "Error:", e)
                    continue

        # Plot all corrected images
        plot_images(corrected_list)
    else:
        exit(1)
