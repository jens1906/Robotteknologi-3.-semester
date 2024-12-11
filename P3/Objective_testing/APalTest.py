import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from P3.main import plot_images

global checkertest
checkertest = False

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from P3.ColorCorrection import ColourCorrectMain as cc

def get_pal_diff(ref_pal, checker, corrected_palette):
    #ref_pal = cv.cvtColor(ref_pal, cv.COLOR_BGR2RGB)
    #checker = cv.cvtColor(checker, cv.COLOR_BGR2RGB)
    #corrected_palette = cv.cvtColor(corrected_palette, cv.COLOR_BGR2RGB)



    ## get palette difference
    print("Getting Checker Difference")
    org_pal = cc.get_color_patches(ref_pal)
    found_pal = cc.get_color_patches(checker)
    cc_pal = cc.get_color_patches(corrected_palette)

    found_pal_diff = (org_pal - found_pal)
    cc_pal_diff = (org_pal - cc_pal)
    #get the sum of each color channel
    #print("Found pal diff", found_pal_diff)
    found_pal_diff_mean = np.mean(found_pal_diff, axis=0).astype(int)
    cc_pal_diff_mean = np.mean(cc_pal_diff, axis=0).astype(int)
    print("Found pal avr diff", found_pal_diff_mean)
    print("cc pal avr diff", cc_pal_diff_mean)

    #make a 3d plot of all points for found diff and cc diff
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(found_pal_diff[:,0], found_pal_diff[:,1], found_pal_diff[:,2], c='b', marker='o')
    ax.scatter(cc_pal_diff[:,0], cc_pal_diff[:,1], cc_pal_diff[:,2], c='r', marker='o')
    #add legend
    ax.legend(['Original - Found', 'Original - Corrected'])


    #add the x,y,z axis line at 0,0,0
    #get the higest positive value of either found or corrected for all 3 channels
    max_diffx = max(max(found_pal_diff[:,0]), max(cc_pal_diff[:,0]))
    max_diffy = max(max(found_pal_diff[:,1]), max(cc_pal_diff[:,1]))
    max_diffz = max(max(found_pal_diff[:,2]), max(cc_pal_diff[:,2]))

    ax.plot([0, max_diffx], [0, 0], [0, 0], c='black')
    ax.plot([0, 0], [0, max_diffy], [0, 0], c='black')
    ax.plot([0, 0], [0, 0], [0, max_diffz], c='black')

    #find the center for both red and blue
    center_found = np.mean(found_pal_diff, axis=0)
    center_cc = np.mean(cc_pal_diff, axis=0)
    #plot the center for both red and blue larger and on top of the other points
    ax.scatter(center_found[0], center_found[1], center_found[2], c='black', marker='x', s=200, label='linear')
    ax.scatter(center_cc[0], center_cc[1], center_cc[2], c='black', marker='x', s=200, label='linearx')
    #plot a line from the center to 0,0,0
    ax.plot([center_found[0], 0], [center_found[1], 0], [center_found[2], 0], c='black')
    ax.plot([center_cc[0], 0], [center_cc[1], 0], [center_cc[2], 0], c='black')

    #print the length of the line from the center to 0,0,0
    #ax.text2D(-0.2, -0.02, "Blue Center to origo: " + str(round(np.linalg.norm(center_found))), transform=ax.transAxes)
    ax.text2D(-0.2, -0.05, "Blue Center: " + str(np.round(center_found)), transform=ax.transAxes)
    ax.text2D(-0.2, -0.1, "Red Center: " + str(np.round(center_cc)), transform=ax.transAxes)

    MSE_found = np.mean(np.square(found_pal_diff), axis=0)
    MSE_cc = np.mean(np.square(cc_pal_diff), axis=0)
    RMSE_found = np.sqrt(MSE_found)
    RMSE_cc = np.sqrt(MSE_cc)
    print("RMSE Found", RMSE_found)
    print("RMSE CC", RMSE_cc)
    RMSE_improvement = (RMSE_found - RMSE_cc) / RMSE_found * 100

    ax.text2D(0.5, -0.1, "RMSE Improvement: " + str(np.round(RMSE_improvement)) + "%", transform=ax.transAxes)

    #ax.text2D(-0.2, -0.12, "Average Improvement: " + str(round(improvement)) + "%", transform=ax.transAxes)

    
    current_color = 'Yellow'

    #set plot title
    ax.set_title(f'RGB Deviation ({current_color} Light): Before vs. After Correction')

    ax.set_xlabel('Deviation in Red')
    ax.set_ylabel('Deviation in Green')
    ax.set_zlabel('Deviation in Blue')

    #save the plot at P3\Results\Data\colcaltest\results as a png with the name *color*_plot.png
    #plt.savefig(f'P3\Results\Data\colcaltest/results/{current_color}_plot.png', bbox_inches='tight')


    plt.show()


    

if checkertest:
    #clear console
    os.system('cls')
    import cv2 as cv
    bad_checker = cv.imread('P3\ColorCorrection\Colour_checker_from_Vikki_Bad.png')
    bad_checker = cv.cvtColor(bad_checker, cv.COLOR_BGR2RGB)

    ref_checker = cv.imread('P3\ColorCorrection\Colour_checker_from_Vikki.png')
    ref_checker = cv.cvtColor(ref_checker, cv.COLOR_BGR2RGB)

    corrected_checker = cc.colour_correct(bad_checker, ref_checker, bad_checker)[2]
    
    checker_diff_corrected = cv.absdiff(ref_checker, corrected_checker)
    checker_diff_bad = cv.absdiff(ref_checker, bad_checker)

    #invert both pictures
    checker_diff_corrected = cv.bitwise_not(checker_diff_corrected)
    checker_diff_bad = cv.bitwise_not(checker_diff_bad)


    #show both
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].imshow(checker_diff_bad)
    axes[0].set_title('Bad Picture Difference')
    axes[0].axis('off')
    
    axes[1].imshow(checker_diff_corrected)
    axes[1].set_title('Corrected Picture Difference')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()