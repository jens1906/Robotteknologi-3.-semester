import sys
sys.path.insert(0, r'C:\Users\Andre\colorcal')
from colorcal import get_color_scheme, calculate_color_correction_matrix, apply_color_correction, load_and_convert_image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_and_convert_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image

def get_color_scheme(board, rows=4, cols=6):
    img_height, img_width, _ = board.shape
    tile_width = img_width // cols
    tile_height = img_height // rows

    rgb_values = []
    for row in range(rows):
        for col in range(cols):
            tile_mid_x = int(col * img_width / cols + (img_width / cols) / 2)
            tile_mid_y = int(row * img_height / rows + (img_height / rows) / 2)
            rgb_value = board[tile_mid_y, tile_mid_x]
            rgb_values.append(rgb_value)
    
    return np.array(rgb_values)

def calculate_color_correction_matrix(source_rgb, target_rgb):
    source_rgb, target_rgb = source_rgb.T, target_rgb.T

    # Step 1: Compute source_rgb * target_rgb^T
    source_target_T = np.dot(source_rgb, target_rgb.T)
    
    # Step 2: Compute target_rgb * target_rgb^T
    target_target_T = np.dot(target_rgb, target_rgb.T)
    
    # Step 3: Take the inverse of target_rgb * target_rgb^T
    target_target_T_inv = np.linalg.inv(target_target_T)
    
    # Step 4: Compute the color correction matrix
    color_correction_matrix = np.dot(source_target_T, target_target_T_inv)
    
    return color_correction_matrix

def apply_color_correction(image, color_correction_matrix):
    # Reshape the image to (num_pixels, 3)
    reshaped_image = image.reshape((-1, 3))
    
    # Apply the color correction matrix
    corrected_image = np.dot(reshaped_image, color_correction_matrix.T)
    
    # Reshape back to the original image shape
    corrected_image = corrected_image.reshape(image.shape)
    
    # Clip values to be in the valid range [0, 255]
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)

    return corrected_image

def main():
    clear_console()

    # Load the images
    normboard = load_and_convert_image('P3/ColorCorrection/Color-Checker.jpg')
    fuckboard = load_and_convert_image('P3/ColorCorrection/Color-Checker-1.png')

    # Get the color schemes for the boards
    normboard_scheme = get_color_scheme(normboard)
    fuckboard_scheme = get_color_scheme(fuckboard)

    # Calculate the color correction matrix
    color_correction_matrix = calculate_color_correction_matrix(normboard_scheme, fuckboard_scheme)

    # Apply the color correction to the original image
    corrected_image = apply_color_correction(normboard, color_correction_matrix)

    # Display the original and corrected images
    f, axarr = plt.subplots(1, 2)  # Creates a 1x2 grid
    axarr[0].imshow(normboard)
    axarr[0].set_title("Original Color Checker")
    axarr[1].imshow(corrected_image)
    axarr[1].set_title("Corrected Color Checker")
    plt.show()
