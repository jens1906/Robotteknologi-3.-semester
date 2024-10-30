import cv2 as cv
import numpy as np
import os

def dark_channel(image, size=15): #size=15
    """Compute the dark channel prior of the image."""
    if len(image.shape) == 3:
        min_channel = np.amin(image, axis=2)
    else:
        min_channel = image  # If the image is already single-channel

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    dark_channel = cv.erode(min_channel, kernel)
    return dark_channel

def atmospheric_light(image, dark_channel):
    """Estimate the atmospheric light in the image."""
    h, w = image.shape[:2]
    num_pixels = h * w
    num_brightest = int(max(num_pixels * 0.001, 1))
    
    dark_vec = dark_channel.ravel()
    image_vec = image.reshape(num_pixels, -1)  # Reshape to (num_pixels, channels)
    
    indices = np.argsort(dark_vec)[-num_brightest:]
    brightest_pixels = image_vec[indices]
    
    A = np.mean(brightest_pixels, axis=0)
    return A

def transmission_map(image, A, omega=0.95, size=15): #sizw=15
    """Estimate the transmission map."""
    norm_image = image / A
    dark_channel_norm = dark_channel(norm_image, size)
    transmission = 1 - omega * dark_channel_norm
    return transmission

def Guidedfilter(im, p, r, eps):
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

def TransmissionRefine(im, et):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)
    return t

def recover_image(image, transmission, A, t0=0.1):
    """Recover the dehazed image."""
    transmission = np.maximum(transmission, t0)
    J = (image - A) / transmission + A
    return J

def dehaze(hazy_image):
    """Main function to dehaze an image."""
    image = hazy_image / 255.0
    
    # Split the image into R, G, B channels
    b_channel, g_channel, r_channel = cv.split(image)
    
    # Compute the dark channel for each channel
    dark_r = dark_channel(r_channel)
    dark_g = dark_channel(g_channel)
    dark_b = dark_channel(b_channel)

    cv.imshow('Dark channel', cv.merge((dark_r, dark_g, dark_b)))
    
    # Estimate the atmospheric light for each channel
    A_r = atmospheric_light(r_channel, dark_r)
    A_g = atmospheric_light(g_channel, dark_g)
    A_b = atmospheric_light(b_channel, dark_b)

    cv.imshow('Atmospheric light', cv.merge((A_r, A_g, A_b)))
    
    # Estimate the transmission map for each channel
    transmission_r = transmission_map(r_channel, A_r)
    transmission_g = transmission_map(g_channel, A_g)
    transmission_b = transmission_map(b_channel, A_b)

    cv.imshow('Transmission map', cv.merge((transmission_r, transmission_g, transmission_b)))

    transmission_refined_r = TransmissionRefine(hazy_image, transmission_r)
    transmission_refined_g = TransmissionRefine(hazy_image, transmission_g)
    transmission_refined_b = TransmissionRefine(hazy_image, transmission_b)

    cv.imshow('Refined transmission', cv.merge((transmission_refined_r, transmission_refined_g, transmission_refined_b)))
    
    # Recover the dehazed image for each channel
    dehazed_r = recover_image(r_channel, transmission_refined_r, A_r)
    dehazed_g = recover_image(g_channel, transmission_refined_g, A_g)
    dehazed_b = recover_image(b_channel, transmission_refined_b, A_b)
    
    # Merge the dehazed channels back into a single image
    dehazed_image = cv.merge((dehazed_b, dehazed_g, dehazed_r))
    
    return dehazed_image

# Example usage
script_dir = os.path.dirname(__file__)
image_path = os.path.join(script_dir, 'Dehaze_Samples', 'city.png')
hazy_image = cv.imread(image_path)
if hazy_image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    dehazed_image = dehaze(hazy_image)
    if dehazed_image is not None:
        cv.imshow('Hazy Image', hazy_image)
        cv.imshow('Dehazed Image', dehazed_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Failed to dehaze the image.")