import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import time

def dark_channel(image, size=15):
    """Compute the dark channel prior of the image."""
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    dark_channel = cv.erode(image, kernel)
    return dark_channel

def underwater_light(image, dark_channel):
    """Estimate the underwater light in the image."""
    h, w = image.shape[:2]
    num_pixels = h * w
    num_brightest = int(max(num_pixels * 0.001, 1))
    
    dark_vec = dark_channel.ravel()
    image_vec = image.reshape(num_pixels, -1)
    
    indices = np.argsort(dark_vec)[-num_brightest:]
    brightest_pixels = image_vec[indices]
    
    A = np.mean(brightest_pixels, axis=0)
    return A

def transmission_map(image, A, omega=0.95, size=15):
    """Estimate the transmission map."""
    norm_image = image / A
    dark_channel_norm = dark_channel(norm_image, size)
    transmission = 1 - omega * dark_channel_norm
    return transmission

def Guidedfilter(im, p, r, eps):
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

def TransmissionRefine(image, estimated_transition_map):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, estimated_transition_map, r, eps)
    return t

def recover_image(image, transmission, A, t0=0.1):
    """Recover the dehazed image."""
    transmission = np.maximum(transmission, t0)
    J = (image - A) / transmission + A
    return J

def dehaze(hazy_image):
    """Main function to dehaze an image."""
    start_time = time.time()
    
    image = hazy_image / 255.0
    channels = cv.split(image)
    dark_channels = []
    A_channels = []
    transmission_maps = []
    refined_transmissions = []
    dehazed_channels = []

    # Timing for each step
    step_times = {}
    
    for channel in channels:
        t0 = time.time()
        dark = dark_channel(channel)
        dark_channels.append(dark)
        step_times['dark_channel'] = time.time() - t0
        
        t0 = time.time()
        A = underwater_light(channel, dark)
        A_channels.append(A)
        step_times['underwater_light'] = time.time() - t0
        
        t0 = time.time()
        transmission = transmission_map(channel, A)
        transmission_maps.append(transmission)
        step_times['transmission_map'] = time.time() - t0
        
        t0 = time.time()
        refined_transmission = TransmissionRefine(hazy_image, transmission)
        refined_transmissions.append(refined_transmission)
        step_times['transmission_refine'] = time.time() - t0
        
        t0 = time.time()
        dehazed = recover_image(channel, refined_transmission, A)
        dehazed_channels.append(dehazed)
        step_times['recover_image'] = time.time() - t0

    dehazed_image = cv.merge(dehazed_channels)

    # Part for adding satuation
    saturation = True
    brightness = True  
    if saturation or brightness:
        dehazed_image = np.clip(dehazed_image * 255, 0, 255).astype(np.uint8)
        dehazed_image = cv.cvtColor(dehazed_image, cv.COLOR_BGR2HSV)
        
        if saturation:
            dehazed_image[:, :, 1] = np.clip(dehazed_image[:, :, 1] * 2.5, 0, 255)  # Adjust saturation
        
        if brightness:
            dehazed_image[:, :, 2] = np.clip(dehazed_image[:, :, 2] * 2.5, 0, 255)  # Adjust brightness (increase factor as needed)
        
        dehazed_image = cv.cvtColor(dehazed_image, cv.COLOR_HSV2BGR)
        dehazed_image = dehazed_image / 255.0  # Normalize back to [0, 1] range

    total_time = time.time() - start_time
    
    # Print timing information
    print("\nDehazing Performance Metrics:")
    print("-" * 30)
    print(f"Total time: {total_time:.3f} seconds")
    print("\nTime per step (averaged across channels):")
    for step, t in step_times.items():
        print(f"{step}: {t/3:.3f} seconds")  # Divide by 3 for RGB channels

    return dehazed_image

def calculate_psnr(image1, image2):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    psnr_value = cv.PSNR(image1, image2)
    return psnr_value

# Example usage
script_dir = os.path.dirname(__file__)
image_path = None #os.path.join(script_dir, 'Dehaze_Samples', 'P3\Results\Data\16th_Milk\Behind_Camera_20241611_120040.png')
hazy_image = cv.imread("P3\Results\Data\Clay\Clay1g\Green_Right_Side_light10_exp73625.0_20242111_115328.png")
if hazy_image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    dehazed_image = dehaze(hazy_image)

    # Convert dehazed image to 8-bit format
    dehazed_image_8bit = np.clip(dehazed_image * 255, 0, 255).astype(np.uint8)

    cv.imwrite('P3\Dehazing\Module test\dehaze.png', dehazed_image_8bit)

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