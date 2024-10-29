import cv2 as cv
import numpy as np
import os

def dark_channel(image):
    size=15
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

def transmission_map(image, A):
    omega=0.95
    size=15
    """Estimate the transmission map."""
    norm_image = image / A
    dark_channel_norm = dark_channel(norm_image, size)
    transmission = 1 - omega * dark_channel_norm
    return transmission

def recover_image(image, transmission, A):
    t0=0.1
    """Recover the scene radiance."""
    transmission = np.maximum(transmission, t0)
    J = (image - A) / transmission[:, :, np.newaxis] + A
    J = np.clip(J, 0, 1)
    return J

def dehaze(hazy_image):
    """Main function to dehaze an image."""
    image = hazy_image / 255.0
    
    # Split the image into R, G, B channels
    b_channel, g_channel, r_channel = cv.split(image)
    
    # Apply dehazing functions on each channel
    dark_r = dark_channel(r_channel)
    dark_g = dark_channel(g_channel)
    dark_b = dark_channel(b_channel)

    A_r = atmospheric_light(r_channel, dark_r)
    A_g = atmospheric_light(g_channel, dark_g)
    A_b = atmospheric_light(b_channel, dark_b)
    
    transmission_r = transmission_map(r_channel, A_r)
    transmission_g = transmission_map(g_channel, A_g)
    transmission_b = transmission_map(b_channel, A_b)
    
    dehazed_r = recover_image(r_channel, transmission_r, A_r)
    dehazed_g = recover_image(g_channel, transmission_g, A_g)
    dehazed_b = recover_image(b_channel, transmission_b, A_b)
    
    # Merge the dehazed channels back into a single image
    dehazed_image = cv.merge((dehazed_b, dehazed_g, dehazed_r))
    
    return dehazed_image

script_dir = os.path.dirname(__file__)
image_path = os.path.join(script_dir, 'Dehaze_Samples', 'underwater.jpg')
hazy_image = cv.imread(image_path)
if hazy_image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    dehazed_image = dehaze(hazy_image)
    cv.imshow('Hazy Image', hazy_image)
    cv.imshow('Dehazed Image', dehazed_image)
    cv.waitKey(0)
    cv.destroyAllWindows()