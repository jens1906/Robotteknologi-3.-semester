import cv2 as cv
import numpy as np
import os
import rawpy
import matplotlib.pyplot as plt

# Den er kode er lidt janky og jeg har ikke fået den til at virke, det er som
# om at der er nogle funktioner der ikke virker optimalt fordi at outputtet er 
# helt blåt, og det er ikke fordi at der er noget galt med billedet, fordi at
# billedet egentlig normalt ser fint ud. Jeg har prøvet at converte til bgr, 
# men forstår ikke :/

def read_raw_image(image_path):
    """Read a raw image using rawpy and convert it to a format OpenCV can handle."""
    with rawpy.imread(image_path) as raw:
        rgb_image = raw.postprocess()
    
    bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
    return rgb_image

def dark_channel(image, size=15):
    """Compute the dark channel prior of the image."""
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    dark_channel = cv.erode(image, kernel)
    return dark_channel

def atmospheric_light(image, dark_channel):
    """Estimate the atmospheric light in the image."""
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

def TransmissionRefine(im, et):
    """Refine the transmission map."""
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 1023.0  # Normalize for 10-bit images
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
    image = hazy_image / 1023.0  # Normalize for 10-bit images
    
    channels = cv.split(image)
    dark_channels = []
    A_channels = []
    transmission_maps = []
    refined_transmissions = []
    dehazed_channels = []

    for channel in channels:
        dark = dark_channel(channel)
        dark_channels.append(dark)
        
        A = atmospheric_light(channel, dark)
        A_channels.append(A)
        
        transmission = transmission_map(channel, A)
        transmission_maps.append(transmission)
        
        refined_transmission = TransmissionRefine(hazy_image, transmission)
        refined_transmissions.append(refined_transmission)
        
        dehazed = recover_image(channel, refined_transmission, A)
        dehazed_channels.append(dehazed)

    dehazed_image = cv.merge(dehazed_channels)
    
    return dehazed_image, dark_channels, transmission_maps, refined_transmissions

def display_images(hazy_image, dehazed_image, dark_channels, transmission_maps, refined_transmissions):
    """Display images using matplotlib."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Resize images for display
    def resize_image(image, width=800):
        height = int(image.shape[0] * (width / image.shape[1]))
        return cv.resize(image, (width, height))
    
    hazy_image_resized = resize_image(hazy_image)
    dehazed_image_resized = resize_image(dehazed_image)
    
    # Convert images to 8-bit for display
    hazy_image_8bit = cv.convertScaleAbs(hazy_image_resized, alpha=(255.0/1023.0))
    dehazed_image_8bit = cv.convertScaleAbs(dehazed_image_resized, alpha=(255.0/1023.0))
    
    # Convert dark channels, transmission maps, and refined transmissions to 8-bit and ensure they have 3 channels
    dark_channel_images_8bit = [cv.cvtColor(cv.convertScaleAbs(resize_image(dark), alpha=(255.0/1023.0)), cv.COLOR_GRAY2BGR) for dark in dark_channels]
    transmission_map_images_8bit = [cv.cvtColor(cv.convertScaleAbs(resize_image(transmission), alpha=(255.0/1023.0)), cv.COLOR_GRAY2BGR) for transmission in transmission_maps]
    refined_transmission_images_8bit = [cv.cvtColor(cv.convertScaleAbs(resize_image(transmission), alpha=(255.0/1023.0)), cv.COLOR_GRAY2BGR) for transmission in refined_transmissions]
    
    # Display the original hazy image
    axes[0, 0].imshow(cv.cvtColor(hazy_image_8bit, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title('Hazy Image')
    axes[0, 0].axis('off')
    
    # Display the dehazed image
    axes[0, 1].imshow(cv.cvtColor(dehazed_image_8bit, cv.COLOR_BGR2RGB))
    axes[0, 1].set_title('Dehazed Image')
    axes[0, 1].axis('off')
    
    # Display the dark channel
    axes[0, 2].imshow(dark_channel_images_8bit[0])
    axes[0, 2].set_title('Dark Channel')
    axes[0, 2].axis('off')
    
    # Display the transmission map
    axes[1, 0].imshow(transmission_map_images_8bit[0])
    axes[1, 0].set_title('Transmission Map')
    axes[1, 0].axis('off')
    
    # Display the refined transmission map
    axes[1, 1].imshow(refined_transmission_images_8bit[0])
    axes[1, 1].set_title('Refined Transmission')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

script_dir = os.path.dirname(__file__)
image_path = os.path.join(script_dir, 'Dehaze_Samples', 'DE haze', 'hazy16.DNG')

# Read the raw image using rawpy
hazy_image = read_raw_image(image_path)

if hazy_image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    dehazed_image, dark_channels, transmission_maps, refined_transmissions = dehaze(hazy_image)
    display_images(hazy_image, dehazed_image, dark_channels, transmission_maps, refined_transmissions)