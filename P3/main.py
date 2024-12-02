
import os
import sys
os.system('cls')
import subprocess

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Input import input as im
from Dehazing import dehaze as dh
from Palette_detection import LocateChecker as lc
from ColorCorrection import ColourCorrectMain as cc
from P3.Objective_testing import APalTest as apt
from P3.Objective_testing import Objective_testing as ot

import math
import time
import inspect
import cv2 as cv
import numpy as np
from vmbpy import *
from datetime import datetime
import matplotlib.pyplot as plt

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    os.system('cls')

try:
    import xlsxwriter
    from openpyxl import load_workbook
except Exception as e:
    print("Error importing xlsxwriter or openpyxl:", e)
    install('openpyxl')
    install('xlsxwriter')

def plot_images(images):
    #if the list is empty raise error
    if len(images) == 0:
        raise ValueError("No images to plot")
    
    #if None in images print the index of None and pop the None
    for i, img in enumerate(images):
        if img is None:
            print(f"None in images at index {i}")
            images.pop(i)
            break  # Exit the loop and re-check, as modifying the list in-place affects iteration

    """
    Plots a given list of images in a grid layout using matplotlib.
    Handles both individual images and lists of images.
    """
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
    if cols > 1:
        rows = math.ceil(n / cols)
    else: 
        return
    
    # Create the plot
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    axs = axs.ravel()  # Flatten the array of axes for easy indexing
    
    for i, img in enumerate(images):
        if img.ndim == 2:  # Grayscale image
            axs[i].imshow(img, cmap='gray')
        else:  # Color image
            axs[i].imshow(img)
        # Use variable names as titles if they exist
        if i < len(image_names):
            axs[i].set_title(image_names[i], fontsize=20)
    
    # Hide any empty subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
    #plt.savefig(f'P3/Results/FullPlots/FullPlot_{timestamp}.png', bbox_inches='tight') #if you want to save the plot

    plt.show()


def main(cam=None, image_path=None, detailed=False):
    # Start time
    start_time = time.perf_counter()

    # Create a list to store images for plotting
    plot_list = []

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
    
    plot_list.append(image)

    ## Dehazing
    print("------Dehazing Image------")
    dehazed_image = dh.dehaze(image)
    if isinstance(dehazed_image, tuple):
        dehazed_image = [np.array(item) for item in dehazed_image]
    plot_list.append(dehazed_image)

    ## Locate Color Checker
    print("------Locating Color Checker------")
    #template = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki_full.png')
    template =  cv.resize(cv.imread("P3\Palette_detection\Colour_checker_from_Vikki_full.png"), (606, 318), interpolation=cv.INTER_AREA)
    #cv.imshow('Template', template)
    #cv.waitKey(0)

    dehazed_checker, corners, wrap_matrix, loc = lc.LocateChecker(dehazed_image, template)
    input_colour_checker, corners, xx, loc = lc.LocateChecker(image, template)
    pre_dehazed_checker = lc.LocateCheckerOriginal(image, template, wrap_matrix)

    ## Color Correction
    print("------Color Correcting Image------")
    original_checker = cv.imread('P3\Palette_detection\Colour_checker_from_Vikki.png')
    original_checker = cv.cvtColor(original_checker, cv.COLOR_BGR2RGB)
    try:
        corrected_image, cc_matrix, corrected_checker = cc.colour_correct(dehazed_image, original_checker, dehazed_checker)
    except:
        raise Exception("Color correction failed")
    
    if detailed:
        #locate checker om corrected image
        corrected_checker_2, corners, wrap_matrix, loc = lc.LocateChecker(corrected_image, template)
        #color correct corrected checker
        corrected_image_2, cc_matrix, corrected_checker_2_pepe = cc.colour_correct(corrected_image, original_checker, corrected_checker_2)
    
    ## Plot Images
    plot_list.append(corrected_image)

    plot_list.append(input_colour_checker)
    plot_list.append(dehazed_checker)

    plot_list.append(corrected_checker)
    plot_list.append(original_checker)    
    if detailed:
        plot_list.append(corrected_checker_2)
        plot_list.append(corrected_image_2)
        pass
        

    print("------Plotting Images------")
    try:
        plot_images(plot_list)
        pass
    except Exception as e:
        print("Error plotting images:", e)

    print("------Finished------")
    return corrected_image, dehazed_image, corrected_checker, pre_dehazed_checker

if __name__ == '__main__':
    cam = None
    image_path = None

    test_method = 'single'  # 'single', 'live', 'folder'

    if test_method == 'single':
        #image_path = 'P3\Results\Data\Gips\Gypsum12g\Beside_Camera_light5_exp112596.0_20242011_145005.png'
        image_path = 'P3\Results\Data\colcaltest/red_beside_light5_exp500005.0_20242611_132609.png'
        #image_path = 'P3\Results\Data\Milk/32th_Milk\Beside_Camera_20241611_121705.png'
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
                        corrected, dehazed, corrected_checker, pre_dehazed_checker = main(cam, None, True)
                        #cv.imshow('Corrected Image', corrected)
                        if cv.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' press
                            break
                    except Exception as e:
                        print("Error during live processing:", e)
                        break
                cv.destroyAllWindows()
    elif test_method == 'folder':
        folder = 'P3\Results\Data\Gips\Gypsum30g'
        os.makedirs(f'{folder}/Results', exist_ok=True)
        #Objective testing excel file
        workbook = xlsxwriter.Workbook(f'{folder}/Results/OTResults.xlsx')
        worksheet = workbook.add_worksheet()
        ot.ReadyExcel(worksheet)
        workbook.close()
        workbook = load_workbook(f'{folder}/Results/OTResults.xlsx')
        worksheet = workbook.active

        corrected_list = []
        for file in os.listdir(folder):
            if file.endswith('.png'):
                image_path = f'{folder}/{file}'
                print("!!!Processing: ", file)
                try:
                    corrected, dehazed, corrected_checker, pre_dehazed_checker = main(cam, image_path)
                    #cv.imshow('Corrected Image', cv.resize(corrected, (0,0), fx=0.5, fy=0.5))
                    #cv.imshow('Dehazed Image', cv.resize(dehazed, (0,0), fx=0.5, fy=0.5))
                    #cv.imshow('Corrected Checker', cv.resize(corrected_checker, (0,0), fx=0.5, fy=0.5))
                    #cv.imshow('Pre Dehazed Checker', cv.resize(pre_dehazed_checker, (0,0), fx=0.5, fy=0.5))
                    #cv.waitKey(0)

                    corrected_list.append(corrected)
                    
                    #Objective Testing
                    ot.ObjectiveTesting(file, corrected, image_path, worksheet, dehazed, corrected_checker, pre_dehazed_checker)         
                    
                    timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
                    cv.imwrite(f'{folder}/Results/{file}_Result_{timestamp}_.png', cv.cvtColor(corrected, cv.COLOR_BGR2RGB))

                except Exception as e:
                    ot.ObjectiveTestingFail(file, worksheet)
                    print("Failed", file, "Error:", e)
                    continue
        
        ot.AdjustExcel(worksheet)
        ot.average(worksheet)
        workbook.save(ExcelFile)

        # Plot all corrected images
        plot_images(corrected_list)
    else:
        exit(1)