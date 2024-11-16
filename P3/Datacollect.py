from Input import input as im
import os
import cv2 as cv
from datetime import datetime

image = im.get_image()
#Remember.. exposure = 20-30k, gain = 0 i API
#image = cv.imread('P3\Results\OrgImages\image_20241311_144042.png')


#   Setup settings
#Exposure 25k, Gain 0
#Light = MAX    Overvej at tage billeder med forskellige lysmængder

#Light positions: 
# Pink = Behind_Camera
# Dark blue = Beside_Camera
# Orange = Right_Side
# Light Green = Left_Side
# Purple = Top_Down
# Light Blue = InFront_Camera

#Water
#1/32 Milk, mixed
#1/16 Milk, mixed
#1/8 Milk, mixed
#1/4 Milk, mixed
#1/2 Milk, mixed
#Full Milk, mixed

#make folder in P3/Results/Data
test_folder = '32th_Milk'
test_name = 'Behind_Camera'

#make folder in P3/Results/Data
try:
    os.makedirs(f'P3/Results/Data/{test_folder}')
except FileExistsError:
    pass

timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
cv.imwrite(f'P3/Results/Data/{test_folder}/{test_name}_{timestamp}.png', cv.cvtColor(image, cv.COLOR_BGR2RGB))