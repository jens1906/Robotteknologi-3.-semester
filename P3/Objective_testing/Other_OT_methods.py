import os
import sys
import subprocess

import cv2
import numpy as np
import math

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    os.system('cls')

try:
    import torch
    import piq
except Exception as e:
    print("Error importing torch or piq:", e)
    install('torch')
    install('piq')

import main

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Palette_detection import LocateChecker as lc



def MSE(imgX, imgY):
    return np.mean((imgX - imgY)**2)   

#Following 2/3 functions is from https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python    
def PSNR(imgX, imgY):
    # img1 and img2 have range [0, 255]
    imgX = imgX.astype(np.float64)
    imgY = imgY.astype(np.float64)
    if MSE(imgX, imgY) == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(MSE(imgX, imgY)))

def ssim(imgX, imgY):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    imgX = imgX.astype(np.float64)
    imgY = imgY.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(imgX, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(imgY, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(imgX**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(imgY**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(imgX * imgY, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def SSIM(imgX, imgY):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not imgX.shape == imgY.shape:
        raise ValueError('Input images must have the same dimensions.')
    if imgX.ndim == 2:
        return ssim(imgX, imgY)
    elif imgX.ndim == 3:
        if imgX.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(imgX, imgY))
            return np.array(ssims).mean()
        elif imgX.shape[2] == 1:
            return ssim(np.squeeze(imgX), np.squeeze(imgY))
    else:
        raise ValueError('Wrong input image dimensions.')
    
def ContrastQualityIndex(imgB, imgA, PatchSize):
    #We want a local but not nested local. Greyscale is not used because of contrast also is affected by colors
    Total_PCQI = 0
    #After this we need to loop through some patches. This means that the picture is set up into smaller areas and calculate these and after this finding the average 
    for y in range(0, imgB.shape[0], PatchSize):
        for x in range(0, imgB.shape[1], PatchSize):
            #Firstly we get the patch area and put this into a array/matrix
            imagePatches = [imgB[y:y+PatchSize, x:x+PatchSize],
                            imgA[y:y+PatchSize, x:x+PatchSize]]
            
            #After this the mean luminance can be found
            imgPatchLuminences = [np.mean(imagePatches[0]),
                                  np.mean(imagePatches[1])]  
            
            #Then the luminance can be compared
            luminenceComparison = (imgPatchLuminences[0] + imgPatchLuminences[1]) / 2 
            
            #Then the product of the comparison can be added to the toal amount
            Total_PCQI += luminenceComparison * imgPatchLuminences[0] * imgPatchLuminences[1]

    #The amount of patches needs to be found by, // is used for a round number. It is rounded because some data can be lost if ImageRes%PatchSice !=0
    Patches = (imgB.shape[0] // PatchSize) * (imgB.shape[1] // PatchSize)

    #Then the average patch contrast/luminence can be found
    avgPacthContrastResult = Total_PCQI / Patches 
    return avgPacthContrastResult


def PatchBasedContrastQualityIndex(imgB, imgA, PatchSize, C=1e-5):
    Total_PCQI = 0
    num_patches = 0

    for y in range(0, imgB.shape[0], PatchSize):
        for x in range(0, imgB.shape[1], PatchSize):
            # Extract patches
            patchB = imgB[y:y+PatchSize, x:x+PatchSize]
            patchA = imgA[y:y+PatchSize, x:x+PatchSize]
            
            # Compute standard deviation (contrast) within each patch
            contrastB = np.std(patchB)
            contrastA = np.std(patchA)
            
            # PCQI calculation for the patch
            pcqi_patch = (2 * contrastB * contrastA + C) / (contrastB**2 + contrastA**2 + C)
            
            # Accumulate PCQI score for all patches
            Total_PCQI += pcqi_patch
            num_patches += 1

    # Average the PCQI over all patches
    avg_PCQI = Total_PCQI / num_patches
    return avg_PCQI


def UnderwaterQualityEvaluation(imgX, imgY):
    #These are some random weighted coefficients, which needs to be found from a reference or chosen
    c = [0.4680,
         0.2745,
         0.2576]

    #For a start the images are converted into LAB format which has
    #L = Luminence
    #A = Color from green to magenta
    #B = Blue to yellow
    imgXCvt = cv2.cvtColor(imgX, cv2.COLOR_BGR2Lab)
    imgYCvt = cv2.cvtColor(imgY, cv2.COLOR_BGR2Lab)

    #This is then split up so it is usable and astype to force it into float
    labValues = [cv2.split(imgXCvt.astype(np.float32)),
                cv2.split(imgYCvt.astype(np.float32))]

    #Chroma can then be found by finding the length of the A and B aka finding the Chroma and then finding the standard deviation by np.std
    Chroma = [np.std(np.sqrt(np.float32(labValues[0][1])**2 + np.float32(labValues[0][2])**2)),
            np.std(np.sqrt(np.float32(labValues[1][1])**2 + np.float32(labValues[1][2])**2))]
    
    #Contrast can in this scenario be found by highest light - lowest light 
    Contrast = [np.max(labValues[0][0]) - np.min(labValues[0][0]),
                np.max(labValues[1][0]) - np.min(labValues[1][0])]
    
    #Now saturation can be found by converting the img to HSV, also called Hue, Saturation, Value(intensity)
    #Then the it is split up and the mean is found of saturation
    AverageSaturation = [np.mean(cv2.split(cv2.cvtColor(imgX, cv2.COLOR_BGR2HSV))[1]),
                        np.mean(cv2.split(cv2.cvtColor(imgY, cv2.COLOR_BGR2HSV))[1])]

    #Then the formula can be calculated
    resultUCIQE = [c[0] * Chroma[0] + c[1] * Contrast[0] + c[2] * AverageSaturation[0],
                   c[0] * Chroma[1] + c[1] * Contrast[1] + c[2] * AverageSaturation[1]]
    return resultUCIQE

def pSSIM(reference,Improved):
    reference_tensor = torch.from_numpy(reference).float() / 255.0
    Improved_tensor = torch.from_numpy(Improved).float() / 255.0

    # Rearrange dimensions to (N, C, H, W)
    reference_tensor = reference_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    Improved_tensor = Improved_tensor.permute(2, 0, 1).unsqueeze(0)    # (1, C, H, W)

    ssim_value = piq.ssim(reference_tensor, Improved_tensor).item() * 100    
    output = f"{round(ssim_value,2)}%"

    return output

def pPSNR(reference,Improved):
    reference_tensor = torch.from_numpy(reference).float() / 255.0
    Improved_tensor = torch.from_numpy(Improved).float() / 255.0

    # Rearrange dimensions to (N, C, H, W)
    reference_tensor = reference_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    Improved_tensor = Improved_tensor.permute(2, 0, 1).unsqueeze(0)    # (1, C, H, W)
    
    return piq.psnr(reference_tensor, Improved_tensor)
def OPSNR(imgX, imgY):

    return round(cv2.PSNR(imgX, imgY), 2)

def MeanBrightnessError(imgX, imgY):
    #In the start the before and after picture get converted into grey scale picture
    #This is done because of greyscale also can be seen at as brightness levels
    imgXGrey = cv2.cvtColor(imgX, cv2.COLOR_BGR2GRAY)
    imgYGrey = cv2.cvtColor(imgY, cv2.COLOR_BGR2GRAY)

    #After the convertion the formula can be calculated for the 2 pictures by doing as seen down under.
    result = round(np.mean(imgXGrey)-np.mean(imgYGrey), 2)
    """
    if result>0:
        output = f"{result} which means the picture has a lower brightness"
    elif result<0:
        output = f"{result} which means the picture has a higher brightness"
    else:
        output = f"There has been no change"
    """        
    return result 


def AverageGradient(imgX, imgY):
    #Firstly the images har converted into greyscale so that there is less information to work with.
    #This can be done because it is wanted to see how big changes there is in the picture, aka sharpness
    #These differences can be found in brightness levels and therefore greyscale is great
    imgXGrey = cv2.cvtColor(imgX, cv2.COLOR_BGR2GRAY)
    imgYGrey = cv2.cvtColor(imgY, cv2.COLOR_BGR2GRAY)

    #The sobel command finds all of the gradients, this works in the format
    #cv2.Sobel(ImageInput, ResolutionOfData, xdirection, ydirection, kernelsize)
    #This means that it here will make a array where it has [x gradient, y gradient]
    gradients = [[cv2.Sobel(imgXGrey, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(imgXGrey, cv2.CV_64F, 0, 1, ksize=3)],
                 [cv2.Sobel(imgYGrey, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(imgYGrey, cv2.CV_64F, 0, 1, ksize=3)]]

    #Now the magnitude of the gradients are found which is done by finding the length of the gradient vectors.
    gradientMagnitutes = [np.sqrt(gradients[0][0]**2 + gradients[0][1]**2),
                          np.sqrt(gradients[1][0]**2 + gradients[1][1]**2)]

    #Then the average gradient can be found
    avgGradientResult = [np.sum(gradientMagnitutes[0])/((imgX.shape[1]-1)*(imgX.shape[0]-1)),
               np.sum(gradientMagnitutes[1])/((imgY.shape[1]-1)*(imgY.shape[0]-1))]
    
    avgGradientResult = np.round(avgGradientResult, decimals=2)
    avgGradientResultCompare = round(avgGradientResult[0]-avgGradientResult[1],2)
    """
    if avgGradientResultCompare>0:
        output = f"{avgGradientResult} which worse by a difference of {abs(avgGradientResultCompare)}"
    elif avgGradientResultCompare<0:
        output = f"{avgGradientResult} which better by a difference of {abs(avgGradientResultCompare)}"
    else:
        output = f"There has been no change"
    """        
    return avgGradientResultCompare

"""
print(f'Mean Square Error: {MSE(imageBefore, imageAfter)}')
print(f'Peak Signal to Noise Ratio: {PSNR(imageBefore, imageAfter)}')
print(f'Structure Similarity Index Method:0 {SSIM(imageBefore, imageAfter)}')
print(f'Mean Brightness Error: {MeanBrightnessError(imageBefore, imageAfter)}')
print(f'Average Gradient: {AverageGradient(imageBefore, imageAfter)}')
print(f'Patch-based Contrast Quality Index (PCQI): {ContrastQualityIndex(imageBefore, imageAfter, 32)}')
print(f'Underwater colour image quality evaluation metric (UCIQE): {UnderwaterQualityEvaluation(imageBefore, imageAfter)}')
print(f'Patch-based Contrast Quality Index (PCQI): {CompareContrastQualityIndex(reference, Improved, 32)} with the value {round(ContrastQualityIndex(reference, Improved, 32),2)}')
"""

#dehazed_checker, corner, warp_matrix, pos = lc.LocateChecker(dehazed_image, template) 

#input_colour_chekcer = lc.LocateCheckerOriginal(image, template, warp_matrix)
def testOfPSNR():
    Template = cv2.imread("P3\Palette_detection\Colour_checker_from_Vikki_full.png")
    #P3\Results\Data\Milk\32th_Milk\Beside_Camera_20241611_121705.png
    original, x2, xx2, xxx2 = lc.LocateChecker(cv2.imread("P3\Results\Data\GroundTruth\Beside_Camera_AutoTarget5_light5_exp29311.0_20242211_103548.png"), Template)
    After, x4, xx4, xxx4 = lc.LocateChecker(cv2.imread("P3\Results\Data\Gips\Gypsum18g\Results\Beside_Camera_light10_exp100281.0_20242011_151124.png_Result_20242111_085609_.png"), Template)
    Before = lc.LocateCheckerOriginal(cv2.imread("P3\Results\Data\Gips\Gypsum18g\Beside_Camera_light10_exp100281.0_20242011_151124.png"), Template, xx4)
    print("Before enhanced: " + str(OPSNR(original, Before)))
    print("After enhanced: " + str(OPSNR(original, After)))
    print("Gauss : " + str(OPSNR(original, cv2.GaussianBlur(After, (25,25),sigmaX=0))))
    #cv2.imshow("No Gauss", After)
    #cv2.imshow("Gauss", cv2.GaussianBlur(After, (15,15),sigmaX=0))
    #cv2.imshow("Median", cv2.medianBlur(After, 5))
    #cv2.waitKey(0)  
    return    

def OLDObjectiveTesting(Improved, reference):
    print(f'Average Gradient: {AverageGradient(reference, Improved)}')
    print(f'Mean Brightness Error: {MeanBrightnessError(reference, Improved)}')
    print(f'Structure Similarity Index Method: {pSSIM(reference, Improved)}')

    #print(f'Structure Similarity Index Method: {SSIM(reference, Improved)}')

def AllObjectiveTesting(Improved, reference):
    print(f'Mean Square Error: {MSE(reference, Improved)}')
    print(f'Peak Signal to Noise Ratio: {cv2.PSNR(reference, Improved)}')
    print(f'Structure Similarity Index Method: {pSSIM(reference, Improved)}')
    print(f'Mean Brightness Error: {MeanBrightnessError(reference, Improved)}')
    print(f'Average Gradient: {AverageGradient(reference, Improved)}')
    print(f'Patch-based Contrast Quality Index (PCQI): {ContrastQualityIndex(reference, Improved, 32)}')
    print(f'Underwater colour image quality evaluation metric (UCIQE): {UnderwaterQualityEvaluation(reference, Improved)}')



"""
ObjectiveTesting(imageAfter, imageBefore)
print("Switch")
ObjectiveTesting(imageBefore, imageBefore)
print("All methods")
AllObjectiveTesting(imageAfter, imageBefore)
"""