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
    height, width = image.shape[:2]
    num_pixels = height * width

    #Number of brightest pixels to consider
    num_brightest_pixels = int(max(num_pixels * 0.001, 1))
    
    #Flatten the dark channel and image
    dark_vector = dark_channel.ravel()
    image_vector = image.reshape(num_pixels, -1)  #Reshape to (num_pixels, channels)
    
    #Get the indices of the brightest pixels in the dark channel
    brightest_indices = np.argsort(dark_vector)[-num_brightest_pixels:]

    #Get the corresponding brightest pixels in the image
    brightest_pixels = image_vector[brightest_indices]
    
    #Compute the atmospheric light as the mean of the brightest pixels
    atmospheric_light = np.mean(brightest_pixels, axis=0)
    return atmospheric_light

def transmission_map(image, atmospheric_light, omega=0.95, size=15):
    """Estimate the transmission map."""
    #Normalize the image by the atmospheric light
    normalized_image = image / atmospheric_light
    #Compute the dark channel of the normalized image
    dark_channel_normalized = dark_channel(normalized_image, size)
    #Estimate the transmission map
    transmission = 1 - omega * dark_channel_normalized
    return transmission

def guided_filter(guidance_image, input_image, radius, epsilon):
    """Apply the guided filter to the image."""
    #Compute the mean of the guidance image and the input image
    mean_guidance = cv.boxFilter(guidance_image, cv.CV_64F, (radius, radius))
    mean_input = cv.boxFilter(input_image, cv.CV_64F, (radius, radius))
    #Compute the mean of the product of the guidance image and the input image
    mean_guidance_input = cv.boxFilter(guidance_image * input_image, cv.CV_64F, (radius, radius))
    #Compute the covariance of the guidance image and the input image
    covariance_guidance_input = mean_guidance_input - mean_guidance * mean_input

    #Compute the variance of the guidance image
    mean_guidance_squared = cv.boxFilter(guidance_image * guidance_image, cv.CV_64F, (radius, radius))
    variance_guidance = mean_guidance_squared - mean_guidance * mean_guidance

    #Compute the linear coefficients a and b
    a = covariance_guidance_input / (variance_guidance + epsilon)
    b = mean_input - a * mean_guidance

    #Compute the mean of the linear coefficients
    mean_a = cv.boxFilter(a, cv.CV_64F, (radius, radius))
    mean_b = cv.boxFilter(b, cv.CV_64F, (radius, radius))

    #Compute the output image
    output_image = mean_a * guidance_image + mean_b
    return output_image

def refine_transmission_map(image, estimated_transmission):
    """Refine the transmission map."""
    #Convert the image to grayscale
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #Normalize the grayscale image
    normalized_grayscale = np.float64(grayscale_image) / 255.0  # Change to 4095 for 12-bit images
    radius = 60
    epsilon = 0.0001
    #Apply the guided filter to refine the transmission map
    refined_transmission = guided_filter(normalized_grayscale, estimated_transmission, radius, epsilon)
    return refined_transmission

def recover_image(image, transmission, atmospheric_light, transmission_threshold=0.1):
    """Recover the dehazed image."""
    #Ensure the transmission map is above a threshold
    transmission = np.maximum(transmission, transmission_threshold)
    #Recover the scene radiance
    recovered_image = (image - atmospheric_light) / transmission + atmospheric_light
    return recovered_image

def dehaze(hazy_image):
    """Main function to dehaze an image."""
    #Normalize the hazy image
    normalized_image = hazy_image / 255.0  #Change to 4095 for 12-bit images
    
    #Split the image into R,G,B channels
    channels = cv.split(normalized_image)
    dark_channels = []
    atmospheric_lights = []
    transmission_maps = []
    refined_transmissions = []
    dehazed_channels = []

    #Process each channel
    for channel in channels:
        #Compute the dark channel
        dark = dark_channel(channel)
        dark_channels.append(dark)
        
        #Estimate the atmospheric light
        atmospheric_light = atmospheric_light(channel, dark)
        atmospheric_lights.append(atmospheric_light)
        
        #Estimate the transmission map
        transmission = transmission_map(channel, atmospheric_light)
        transmission_maps.append(transmission)
        
        #Refine the transmission map
        refined_transmission = refine_transmission_map(hazy_image, transmission)
        refined_transmissions.append(refined_transmission)
        
        #Recover the dehazed image
        dehazed = recover_image(channel, refined_transmission, atmospheric_light)
        dehazed_channels.append(dehazed)
    
    #Merge the dehazed channels back into a single image
    dehazed_image = cv.merge(dehazed_channels)
    
    return dehazed_image