from Input import input as im
import os
import cv2 as cv
from datetime import datetime

image = im.get_image()
#Remember.. exposure = 20-30k, gain = 0 i API
#image = cv.imread('P3\Results\OrgImages\image_20241311_144042.png')



#Light positions: 
# Pink = Behind_Camera 
# Dark blue = Beside_Camera
# Orange = Right_Side
# Light Green = Left_Side
# Purple = Top_Down
# Light Blue = InFront_Camera

#Water
#1/2 Milk
#Milk

#make folder in P3/Results/Data
test_folder = 'Water'
test_name = 'Beside_Camera'

#make folder in P3/Results/Data
try:
    os.makedirs(f'P3/Results/Data/{test_folder}')
except FileExistsError:
    pass

timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
cv.imwrite(f'P3/Results/Data/{test_folder}/{test_name}_{timestamp}.png', image)