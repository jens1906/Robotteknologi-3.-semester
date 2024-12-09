import cv2
import image_dehazer										# Load the library
import os
from ColorCorrectionML import ColorCorrectionML


img = cv2.imread('Results/Data/Clay/Clay1g/Beside_Camera_light5_exp67607.0_20242111_115120.png')

cc = ColorCorrectionML(img, chart='Classic', illuminant='D50')

method = 'pls' # 'linear', 'lstsq', 'pls' 
# for linear regression, least square regression, and partial least square regression respectively
show = True

kwargs = {
    'method': method,
    'degree': 3, # degree of polynomial
    'interactions_only': False, # only interactions terms,
    'ncomp': 10, # number of components for PLS only
    'max_iter': 5000, # max iterations for PLS only
    'white_balance_mtd': 0 # 0: no white balance, 1: learningBasedWB, 2: simpleWB, 3: grayWorldWB,
    }

M, patch_size = cc.compute_correction(
    show=show,
    **kwargs
)
    

# resize img by 2
# img = cv2.resize(img, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

img_corr = cc.correct_img(img, show=True)
# img_corr = cc.Parallel_correct_img(img, chunks_=50000, show=True)




"""
# folder path
folder = 'Results\Data\Clay\Clay1g'

original_list = []
corrected_list = []
for file in os.listdir(folder):
    if file.endswith('.png'):
        image_path = f'{folder}/{file}'
        print("!!!Processing: ", file)
        try:
            image = cv2.imread(image_path)
            HazeCorrectedImg, HazeMap = image_dehazer.remove_haze(image)		# Remove Haze
            cv2.destroyAllWindows()
            original_list.append(image)
            corrected_list.append(HazeCorrectedImg)

        except Exception as e:
            print("Failed", file, "Error:", e)
            continue

print("Printing the images")

for i in range(len(original_list)):
    cv2.imshow('original_image', original_list[i]);			# display the original image
    cv2.imshow('enhanced_image', corrected_list[i]);			# display the result
    cv2.waitKey(0)				"""					# hold the display window

