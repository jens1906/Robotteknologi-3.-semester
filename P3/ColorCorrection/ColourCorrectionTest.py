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

def get_color_patches(image, rows=4, cols=6, patch_size=10):
    img_height, img_width, _ = image.shape
    tile_width = img_width // cols
    tile_height = img_height // rows

    rgb_values = []
    for row in range(rows):
        for col in range(cols):
            tile_x = int(col * tile_width + tile_width / 2)
            tile_y = int(row * tile_height + tile_height / 2)
            
            # Extract a patch around the central pixel
            patch = image[tile_y - patch_size//2 : tile_y + patch_size//2,
                          tile_x - patch_size//2 : tile_x + patch_size//2]
            avg_color = np.mean(patch, axis=(0, 1))  # Mean RGB value of the patch
            rgb_values.append(avg_color)
    
    return np.array(rgb_values)

def calculate_color_correction_matrix(source_rgb, target_rgb):
    """Calculate the color correction matrix using least squares method."""
    # Reshape matrices to have samples as rows and RGB values as columns
    source_rgb = source_rgb.reshape(-1, 3)  # Shape: (24, 3)
    target_rgb = target_rgb.reshape(-1, 3)  # Shape: (24, 3)
    
    # Calculate the color correction matrix using least squares
    # X @ CCM = Y, where X is target_rgb, Y is source_rgb
    # CCM = (X^T @ X)^-1 @ X^T @ Y
    X = target_rgb
    Y = source_rgb
    
    XtX = np.dot(X.T, X)  # Shape: (3, 3)
    XtX_inv = np.linalg.inv(XtX)  # Shape: (3, 3)
    XtY = np.dot(X.T, Y)  # Shape: (3, 3)
    
    color_correction_matrix = np.dot(XtX_inv, XtY)  # Shape: (3, 3)
    
    return color_correction_matrix

def apply_color_correction(image, color_correction_matrix):
    # Reshape the image to (num_pixels, 3)
    reshaped_image = image.reshape((-1, 3))
    
    # Apply the color correction matrix
    corrected_image = np.dot(reshaped_image, color_correction_matrix)
    
    # Reshape back to the original image shape
    corrected_image = corrected_image.reshape(image.shape)
    
    # Clip values to be in the valid range [0, 255]
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    
    return corrected_image


def main():
    clear_console()
    print("Color Correction Testing: True")
    print("---------------Import Test Boards--------------")
    
    reference_image_path = 'P3/ColorCorrection/Color-Checker.jpg'
    target_image_path = 'P3/ColorCorrection/U_Water_Sim_ColourUltimate.png'
    
    reference_image = load_and_convert_image(reference_image_path)
    target_image = load_and_convert_image(target_image_path)

    print("---------------Extract Color Patches--------------")
    reference_patches = get_color_patches(reference_image)
    target_patches = get_color_patches(target_image)
    
    print("Reference Patches:")
    print(reference_patches)
    print("Target Patches:")
    print(target_patches)
    
    print("---------------Calculate Color Correction Matrix--------------")
    color_correction_matrix = calculate_color_correction_matrix(reference_patches, target_patches)
    
    print("Color Correction Matrix (Mcc):")
    print(color_correction_matrix)
    
    print("---------------Apply Color Correction--------------")
    corrected_image = apply_color_correction(target_image, color_correction_matrix)
    
    # Display the reference, original, and corrected images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Reference Image')
    plt.imshow(reference_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Original Image')
    plt.imshow(target_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Corrected Image')
    plt.imshow(corrected_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
