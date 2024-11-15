from Input import input as im

import os
import cv2 as cv
import vmbpy as vmb
from datetime import datetime

#image = im.get_image()
image = cv.imread('P3\Results\OrgImages\image_20241311_144042.png')
#make folder in P3/Results/Data
test_folder = 'maelk'
test_name = 'forfra'


#make folder in P3/Results/Data
try:
    os.makedirs(f'P3/Results/Data/{test_folder}')
except FileExistsError:
    pass

timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
cv.imwrite(f'P3/Results/Data/{test_folder}/{test_name}_{timestamp}.png', image)