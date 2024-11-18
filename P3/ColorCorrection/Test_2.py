import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_color_scheme(board):
    rows, cols = 4, 6
    imgheight, imgwidth, dim = board.shape
    tile_width = imgwidth // cols
    tile_height = imgheight // rows

    rgb_list = []
    for r in range(rows):
        for c in range(cols):
            tilemidx = int(c * imgwidth / cols + (imgwidth / cols) / 2)
            tilemidy = int(r * imgheight / rows + (imgheight / rows) / 2)
            rgb_value = board[tilemidy, tilemidx]
            rgb_list.append(rgb_value)
    
    return np.array(rgb_list) / 255.0  # Normalize to [0, 1]

def calculate_mcc(SRGB, IRGB):
    SRGB, IRGB = SRGB.T, IRGB.T

    # Step 1: Compute SRGB * IRGB^T
    Srgb_Irgb_T = np.dot(SRGB, IRGB.T)
    
    # Step 2: Compute IRGB * IRGB^T
    Irgb_Irgb_T = np.dot(IRGB, IRGB.T)
    
    # Step 3: Take the inverse of IRGB * IRGB^T
    Irgb_Irgb_T_inv = np.linalg.inv(Irgb_Irgb_T)
    
    # Step 4: Compute M_CC
    M_CC = np.dot(Srgb_Irgb_T, Irgb_Irgb_T_inv)
    
    return M_CC

def apply_color_correction(image, M_CC):
    # Reshape the image to a 2D array of pixels
    h, w, c = image.shape
    image_reshaped = image.reshape(-1, c) / 255.0  # Normalize to [0, 1]
    
    # Apply the color correction matrix
    corrected_image = np.dot(image_reshaped, M_CC.T)
    
    # Reshape back to the original image shape
    corrected_image = corrected_image.reshape(h, w, c)
    
    # Clip values to be in the valid range [0, 1] and convert to uint8
    corrected_image = np.clip(corrected_image, 0, 1) * 255.0
    corrected_image = corrected_image.astype(np.uint8)
    
    return corrected_image

def visualize_colors(colors, title):
    fig, ax = plt.subplots(1, len(colors), figsize=(15, 2))
    for i, color in enumerate(colors):
        ax[i].imshow([[color]])
        ax[i].axis('off')
    plt.suptitle(title)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load the image of the color checker board
    board = cv2.imread('P3\\ColorCorrection\\Color-Checker-1.png')
    if board is None:
        raise FileNotFoundError("The image file 'Color-Checker-1.png' was not found.")
    
    # Get the color scheme from the board (IRGB)
    IRGB = get_color_scheme(board)
    
    # Load the reference image of the color checker board
    reference_board = cv2.imread('P3\\ColorCorrection\\Color-Checker.jpg')
    if reference_board is None:
        raise FileNotFoundError("The image file 'Color-Checker.jpg' was not found.")
    
    # Get the color scheme from the reference board (SRGB)
    SRGB = get_color_scheme(reference_board)
    
    # Visualize the extracted colors
    visualize_colors(IRGB, "Extracted Colors from Board (IRGB)")
    visualize_colors(SRGB, "Extracted Colors from Reference Board (SRGB)")
    
    # Calculate the color correction matrix
    M_CC = calculate_mcc(SRGB, IRGB)
    
    # Apply the color correction to the image
    corrected_image = apply_color_correction(board, M_CC)
    
    # Save the corrected image
    cv2.imwrite('P3\\ColorCorrection\\corrected_image.jpg', corrected_image)