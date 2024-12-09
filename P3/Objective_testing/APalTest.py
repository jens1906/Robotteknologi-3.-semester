import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
global checkertest
checkertest = False

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from P3.ColorCorrection import ColourCorrectMain as cc

def get_pal_diff(ref_pal, checker, corrected_palette):
    ref_pal = cv.cvtColor(ref_pal, cv.COLOR_BGR2RGB)
    checker = cv.cvtColor(checker, cv.COLOR_BGR2RGB)
    corrected_palette = cv.cvtColor(corrected_palette, cv.COLOR_BGR2RGB)




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
    #write underneath the plot that the blue is the found and red is the corrected
    ax.text2D(-0.2, -0.05, "Blue = Original - Found", transform=ax.transAxes)
    ax.text2D(-0.2, -0.1, "Red = Original - Corrected", transform=ax.transAxes)
    #add the x,y,z axis line at 0,0,0
    #get the higest positive value of either found or corrected for all 3 channels
    max_diffx = max(max(found_pal_diff[:,0]), max(cc_pal_diff[:,0]))
    max_diffy = max(max(found_pal_diff[:,1]), max(cc_pal_diff[:,1]))
    max_diffz = max(max(found_pal_diff[:,2]), max(cc_pal_diff[:,2]))

    ax.plot([0, max_diffx], [0, 0], [0, 0], c='black')
    ax.plot([0, 0], [0, max_diffy], [0, 0], c='black')
    ax.plot([0, 0], [0, 0], [0, max_diffz], c='black')


    #set plot title
    ax.set_title('(Yellow Light) Deviation in RGB Space Before and After Correction')

    ax.set_xlabel('Difference in Red')
    ax.set_ylabel('Difference in Green')
    ax.set_zlabel('Difference in Blue')

    #save the plot at P3\Results\Data\colcaltest\results as a png with the name *color*_plot.png
    #plt.savefig('P3\Results\Data\colcaltest/results/yellow_plot.png')


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