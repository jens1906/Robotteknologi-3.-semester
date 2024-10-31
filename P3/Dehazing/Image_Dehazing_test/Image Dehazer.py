import cv2 as cv
import numpy as np
import os

def dark_channel(image, size=15):
    """Compute the dark channel prior of the image."""
    #Create a morphological kernel
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))

    #Apply erosion to the minimum channel to get the dark channel
    dark_channel = cv.erode(image, kernel)
    return dark_channel

def atmospheric_light(image, dark_channel):
    """Estimate the atmospheric light in the image."""
    #Get the dimensions of the image
    h, w = image.shape[:2]
    num_pixels = h * w

    #Number of brightest pixels to consider
    num_brightest = int(max(num_pixels * 0.001, 1))
    
    #Flatten the dark channel and image
    dark_vec = dark_channel.ravel()
    image_vec = image.reshape(num_pixels, -1)  #Reshape to (num_pixels, channels)
    
    #Get the indices of the brightest pixels in the dark channel
    indices = np.argsort(dark_vec)[-num_brightest:]

    #Get the corresponding brightest pixels in the image
    brightest_pixels = image_vec[indices]
    
    #Compute the atmospheric light as the mean of the brightest pixels
    A = np.mean(brightest_pixels, axis=0)
    return A

def transmission_map(image, A, omega=0.95, size=15):
    """Estimate the transmission map."""
    #Normalize the image by the atmospheric light
    norm_image = image / A
    #Compute the dark channel of the normalized image
    dark_channel_norm = dark_channel(norm_image, size)
    #Estimate the transmission map
    transmission = 1 - omega * dark_channel_norm
    return transmission

def Guidedfilter(im, p, r, eps):
    """Apply the guided filter to the image."""
    #Compute the mean of the guidance image and the input image
    mean_I = cv.boxFilter(im, cv.CV_64F, (r, r))
    mean_p = cv.boxFilter(p, cv.CV_64F, (r, r))
    #Compute the mean of the product of the guidance image and the input image
    mean_Ip = cv.boxFilter(im * p, cv.CV_64F, (r, r))
    #Compute the covariance of the guidance image and the input image
    cov_Ip = mean_Ip - mean_I * mean_p

    #Compute the variance of the guidance image
    mean_II = cv.boxFilter(im * im, cv.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    #Compute the linear coefficients a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    #Compute the mean of the linear coefficients
    mean_a = cv.boxFilter(a, cv.CV_64F, (r, r))
    mean_b = cv.boxFilter(b, cv.CV_64F, (r, r))

    #Compute the output image
    q = mean_a * im + mean_b
    return q

def TransmissionRefine(im, et):
    """Refine the transmission map."""
    #Convert the image to grayscale
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #Normalize the grayscale image
    gray = np.float64(gray) / 255.0  #Change to 4095 for 12-bit images
    r = 60
    eps = 0.0001
    #Apply the guided filter to refine the transmission map
    t = Guidedfilter(gray, et, r, eps)
    return t

def recover_image(image, transmission, A, t0=0.1):
    """Recover the dehazed image."""
    #Ensure the transmission map is above a threshold
    transmission = np.maximum(transmission, t0)
    #Recover the scene radiance
    J = (image - A) / transmission + A
    return J

def dehaze(hazy_image):
    """Main function to dehaze an image."""
    #Normalize the hazy image
    image = hazy_image / 255.0  #Change to 4095 for 12-bit images
    
    #Split the image into R, G, B channels
    channels = cv.split(image)
    dark_channels = []
    A_channels = []
    transmission_maps = []
    refined_transmissions = []
    dehazed_channels = []

    #Process each channel
    for channel in channels:
        #Compute the dark channel for the given channel
        dark = dark_channel(channel)
        dark_channels.append(dark)
        
        #Estimate the atmospheric light for the given channel
        A = atmospheric_light(channel, dark)
        A_channels.append(A)
        
        #Estimate the transmission map for the given channel 
        transmission = transmission_map(channel, A)
        transmission_maps.append(transmission)
        
        #Refine the transmission map for the given channel
        refined_transmission = TransmissionRefine(hazy_image, transmission)
        refined_transmissions.append(refined_transmission)
        
        #Recover the dehazed image for the given channel
        dehazed = recover_image(channel, refined_transmission, A)
        dehazed_channels.append(dehazed)

    #Display intermediate results (delete when implemented in the main script!!)
    cv.imshow('Dark channel', cv.merge(dark_channels))
    cv.imshow('Transmission map', cv.merge(transmission_maps))
    cv.imshow('Refined transmission', cv.merge(refined_transmissions))

    #Merge the dehazed channels back into a single image
    dehazed_image = cv.merge(dehazed_channels)

    output_dir = os.path.join(script_dir, 'Dehaze_Results')
    cv.imwrite(os.path.join(output_dir, 'dark_channel.png'), cv.merge(dark_channels)*255)
    cv.imwrite(os.path.join(output_dir, 'transmission_map.png'), cv.merge(transmission_maps)*255)
    cv.imwrite(os.path.join(output_dir, 'refined_transmission.png'), cv.merge(refined_transmissions)*255)
    cv.imwrite(os.path.join(output_dir, 'dehazed_image.png'), dehazed_image*255)
       
    return dehazed_image

#Example usage (delete when implemented in the main script!!)
script_dir = os.path.dirname(__file__)
image_path = os.path.join(script_dir, 'Dehaze_Samples', 'underwater.jpg')
hazy_image = cv.imread(image_path)
if hazy_image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    dehazed_image = dehaze(hazy_image)
    cv.imshow('Dehazed image', dehazed_image)
    cv.waitKey(0)
    cv.destroyAllWindows()