from Input import input as im
import os
import cv2 as cv
from datetime import datetime

exposure = 278830
image = im.get_image(exposure)
#Remember.. exposure = 20-30k, gain = 0 i API
#image = cv.imread('P3\Results\OrgImages\image_20241311_144042.png')


#   Setup settings
#Exposure 25k, Gain 0
#Light = 5,10    Overvej at tage billeder med forskellige lysmængder
#Exposure target = 20
#Exposure = Auto

#Light positions: 
# Pink = Underwater_Beside_Camera
# Dark blue = Beside_Camera
# Orange = Right_Side
# Light Blue = InFront_Camera




import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()

#make folder in P3/Results/Data
test_folder = 'Gypsum15g'
test_name = 'Underwater_Beside_Camera'
light_intensity = 'x'

#make folder in P3/Results/Data
try:
    os.makedirs(f'P3/Results/Data/{test_folder}')
except FileExistsError:
    pass

timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
cv.imwrite(f'P3/Results/Data/{test_folder}/{test_name}_light{light_intensity}_exp{im.get_exposure()}_{timestamp}.png', cv.cvtColor(image, cv.COLOR_BGR2RGB))