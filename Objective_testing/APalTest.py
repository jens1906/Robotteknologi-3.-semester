import os
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from P3.ColorCorrection import ColourCorrectMain as cc

def get_pal_diff(ref_pal, checker, corrected_palette):
    ## get palette difference
    print("------Getting Palette Difference--")
    org_pal = cc.get_color_patches(ref_pal)
    found_pal = cc.get_color_patches(checker)
    cc_pal = cc.get_color_patches(corrected_palette)
    print(org_pal[0])
    print(found_pal[0])
    print(cc_pal[0])
    print("------")
    found_pal_diff = (org_pal - found_pal)
    cc_pal_diff = (org_pal - cc_pal)
    #get the sum of each color channel
    found_pal_diff = np.mean(found_pal_diff, axis=0).astype(int)
    cc_pal_diff = np.mean(cc_pal_diff, axis=0).astype(int)
    print("Found pal avr diff", found_pal_diff)
    print("cc pal avr diff", cc_pal_diff)

    #make a 3d plot of all points for found diff and cc diff
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(found_pal[:,0], found_pal[:,1], found_pal[:,2], c='r', marker='o')
    ax.scatter(cc_pal[:,0], cc_pal[:,1], cc_pal[:,2], c='b', marker='o')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    plt.show()