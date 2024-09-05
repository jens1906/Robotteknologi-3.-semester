from __future__ import print_function
from builtins import input
from numpy.linalg import norm
import os
import random
import cv2 as cv
import numpy as np

def get_image_brightness(image):
    if len(image.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return round(np.average(norm(image, axis=2)) / np.sqrt(3))
    else:
        # Grayscale
        return round(np.average(image))

def get_random_ExDark_image_path():
    base_path = "LightEnhance\ExDark"
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    random_folder = random.choice(folders)
    folder_path = os.path.join(base_path, random_folder)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    random_image = random.choice(files)
    image_path = os.path.join(folder_path, random_image)
    return image_path

image_path = get_random_ExDark_image_path()
print("Image path: ", image_path)
image = cv.imread(image_path)
#image = cv.imread("LightEnhance/ExDark/Bottle/2015_01341.jpg")
#image = cv.imread(cv.samples.findFile(args.input))

if image is None:
    print('Could not open or find the image: Exiting')
    exit(0)

new_image = np.zeros(image.shape, image.dtype)
 
alpha = 1.0 
beta = 0    

print('-------------------------')
try:
    alpha = float(input('* Contrast modifier [1.0-3.0]: '))
    beta = int(input('* Brightness modifier [0-100]: '))
except ValueError:
    print('Error, not a number')
print("Image shape: ", image.shape)

print("Image brightness value: ", get_image_brightness(image))

 
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

print("New image brightness value: ", get_image_brightness(new_image))

cv.imshow('Original Image', image)
cv.imshow('New Image', new_image)
 
cv.waitKey()