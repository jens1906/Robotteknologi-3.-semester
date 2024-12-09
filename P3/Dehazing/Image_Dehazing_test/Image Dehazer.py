import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import time

def dark_channel(image, size=15):
    """Compute the dark channel prior of the image."""
    min_channel = np.min(image, axis=2)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    dark_channel = cv.erode(min_channel, kernel)
    cv.imshow('dark_channel', dark_channel)
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
    print(f"Atmospheric light: {A}")
    return A

def transmission_map(image, A, omega=0.95, size=15):
    """Estimate the transmission map."""
    norm_image = image / A
    dark_channel_norm = dark_channel(norm_image, size)
    transmission = 1 - omega * dark_channel_norm
    cv.imshow('transmission', transmission)
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
    cv.imshow('refined_transmission', refined_transmission)
    return refined_transmission

def recover_image(image, transmission, A, transmission_threshold=0.1):
    """Recover the dehazed image."""
    transmission = np.maximum(transmission, transmission_threshold)
    transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)  # Broadcast transmission to match image shape
    J = (image - A) / transmission + A
    return J

def dehaze(hazy_image):
    """Main function to dehaze an image."""
    start_time = time.time()
    
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
    
    total_time = time.time() - start_time
    
    # Print timing information
    print("\nDehazing Performance Metrics:")
    print("-" * 30)
    print(f"Total time: {total_time:.3f} seconds")

    return dehazed_image

def calculate_psnr(image1, image2):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    psnr_value = cv.PSNR(image1, image2)
    return psnr_value

# Example usage
script_dir = os.path.dirname(__file__)
image_path = "P3/Results/Data/Clay/Clay1g/Green_Beside_Camera_light5_exp111285.0_20242111_115220.png"
hazy_image = cv.imread(image_path)
if hazy_image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    dehazed_image = dehaze(hazy_image)

    # Convert dehazed image to 8-bit format
    dehazed_image_8bit = np.clip(dehazed_image * 255, 0, 255).astype(np.uint8)

    cv.imwrite('P3/Dehazing/Module test/dehaze.png', dehazed_image_8bit)

    plt.figure(figsize=(10, 5))
        
    plt.subplot(1, 2, 1)
    plt.title('Hazy Image')
    plt.imshow(cv.cvtColor(hazy_image, cv.COLOR_BGR2RGB))
    plt.axis('off')
        
    plt.subplot(1, 2, 2)
    plt.title('Dehazed Image')
    plt.imshow(cv.cvtColor(dehazed_image_8bit, cv.COLOR_BGR2RGB))
    plt.axis('off')
        
    plt.tight_layout()
    plt.show()