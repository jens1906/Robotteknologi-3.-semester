from Input import input as im
import os
import cv2 as cv
from datetime import datetime

exposure = 500005
image = im.get_image(exposure)

import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()

#make folder in P3/Results/Data
test_folder = 'colcaltest'
test_name = 'purple_beside'
light_intensity = '10'

#make folder in P3/Results/Data
try:
    os.makedirs(f'P3/Results/Data/{test_folder}')
except FileExistsError:
    pass

timestamp = datetime.now().strftime("%Y%d%m_%H%M%S")
cv.imwrite(f'P3/Results/Data/{test_folder}/{test_name}_light{light_intensity}_exp{im.get_exposure()}_{timestamp}.png', cv.cvtColor(image, cv.COLOR_BGR2RGB))



#Remember.. exposure = 20-30k, gain = 0 i API
#image = cv.imread('P3\Results\OrgImages\image_20241311_144042.png')


#   Setup settings
#Gain 0
#Light = 5,10  
#Exposure target = 20
#Exposure = Auto/Continuous

#Light positions: 
# Pink = Underwater_Beside_Camera
# Dark blue = Beside_Camera
# Orange = Right_Side
# Light Blue = InFront_Camera
