import cv2 as cv
import numpy as np

def dark_channel(image, size=15): # size = 15
    """Compute the dark channel prior of the image."""
    min_channel = np.amin(image, axis=2)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    dark_channel = cv.erode(min_channel, kernel)
    return dark_channel

def atmospheric_light(image, dark_channel):
    """Estimate the atmospheric light in the image."""
    h, w = image.shape[:2]
    num_pixels = h * w
    num_brightest = int(max(num_pixels * 0.001, 1))
    
    dark_vec = dark_channel.ravel()
    image_vec = image.reshape(num_pixels, 3)
    
    indices = np.argsort(dark_vec)[-num_brightest:]
    brightest_pixels = image_vec[indices]
    
    A = np.mean(brightest_pixels, axis=0)
    return A

def transmission_map(image, A, omega=0.95, size=15): # omega = 0.95, size = 15
    """Estimate the transmission map."""
    norm_image = image / A
    dark_channel_norm = dark_channel(norm_image, size)
    transmission = 1 - omega * dark_channel_norm
    return transmission

def recover_image(image, transmission, A, t0=0.1): # t0 = 0.1
    """Recover the scene radiance."""
    transmission = np.maximum(transmission, t0)
    J = (image - A) / transmission[:, :, np.newaxis] + A
    J = np.clip(J, 0, 1)
    return J

def dehaze(image_path):
    """Main function to dehaze an image."""
    image = cv.imread(image_path) / 255.0
    
    dark = dark_channel(image)
    A = atmospheric_light(image, dark)
    transmission = transmission_map(image, A)
    
    dehazed_image = recover_image(image, transmission, A)
    
    return (dehazed_image * 255).astype(np.uint8)


image_path = r'P3\Dehazing\Image Dehazing test\Dehaze_Samples\forest.jpg'

# Original image
cv.namedWindow("output", cv.WINDOW_NORMAL)   
im = cv.imread(image_path) 
im = dehaze(image_path)                 
imS = cv.resize(im, (960, 540))                
cv.imshow("output", imS)                      

# Dehazed image
cv.namedWindow("original", cv.WINDOW_NORMAL)   
im = cv.imread(image_path)                  
imS = cv.resize(im, (960, 540))                
cv.imshow("original", imS)                    

cv.waitKey(0)




