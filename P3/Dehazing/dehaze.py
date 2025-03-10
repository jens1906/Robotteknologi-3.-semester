import cv2 as cv
import numpy as np
import os

def dark_channel(image, size=15):
    """Compute the dark channel prior of the image."""
    min_channel = np.min(image, axis=2)
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

def transmission_map(image, A, size=15):
    """Estimate the transmission map."""
    norm_image = image / A
    dark_channel_norm = dark_channel(norm_image, size)
    transmission = 1 - dark_channel_norm
    return transmission

def guided_filter(im, p, r, eps):
    """Apply the guided filter to the image."""
    mean_I = cv.boxFilter(im, cv.CV_64F, (r, r))
    mean_p = cv.boxFilter(p, cv.CV_64F, (r, r))
    mean_Ip = cv.boxFilter(im * p, cv.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv.boxFilter(im * im, cv.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv.boxFilter(a, cv.CV_64F, (r, r))
    mean_b = cv.boxFilter(b, cv.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q

def refine_transmission_map(image, estimated_transmission):
    """Refine the transmission map."""
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    normalized_grayscale = np.float64(grayscale_image) / 255.0
    radius = 60
    epsilon = 0.0001
    refined_transmission = guided_filter(normalized_grayscale, estimated_transmission, radius, epsilon)
    return refined_transmission

def recover_image(image, transmission, A, transmission_threshold=0.1):
    """Recover the dehazed image."""
    transmission = np.maximum(transmission, transmission_threshold)
    transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)  # Broadcast transmission to match image shape
    J = (image - A) / transmission + A
    return J

def dehaze(hazy_image):
    """Main function to dehaze an image."""
    
    image = hazy_image / 255.0
    
    # Compute the dark channel
    dark = dark_channel(image)
    
    # Estimate the atmospheric light
    A = atmospheric_light(image, dark)
    
    # Estimate the transmission map
    transmission = transmission_map(image, A)
    
    # Refine the transmission map
    refined_transmission = refine_transmission_map(hazy_image, transmission)
    
    # Recover the dehazed image
    dehazed_image = recover_image(image, refined_transmission, A)

    return np.clip(dehazed_image * 255, 0, 255).astype(np.uint8)