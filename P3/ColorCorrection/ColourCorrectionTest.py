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
    print("Color Correction Testing: True")
    print("---------------Import Test Boards--------------")
    
    original_board_path = 'P3/ColorCorrection/Color-Checker.jpg'
    modified_board_path = 'P3/ColorCorrection/U_Water_Sim_Colour.png'
    
    original_board = load_and_convert_image(original_board_path)
    modified_board = load_and_convert_image(modified_board_path)

    print("---------------Test calculate_color_correction_matrix--------------")
    source_rgb = get_color_scheme(original_board)
    target_rgb = get_color_scheme(modified_board)
    
    print("Source RGB Values:")
    print(source_rgb)
    print("Target RGB Values:")
    print(target_rgb)
    
    color_correction_matrix = calculate_color_correction_matrix(source_rgb, target_rgb)
    
    print("Color Correction Matrix (Mcc):")
    print(color_correction_matrix)
    
    adjusted_target_rgb = np.dot(color_correction_matrix, target_rgb.T).T
    print("Adjusted Target RGB Scheme:")
    print(adjusted_target_rgb)
    
    # Apply the color correction matrix to the entire image
    corrected_image = apply_color_correction(modified_board, color_correction_matrix)
    
    # Display the reference, original, and corrected images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Reference Image')
    plt.imshow(original_board)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Original Image')
    plt.imshow(modified_board)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Corrected Image')
    plt.imshow(corrected_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()