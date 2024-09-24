import cv2
import numpy as np

# Load the image
imageBefore = cv2.imread('Objective testing before.jpg')
imageAfter = cv2.imread('Objective testing after.jpg')


def MeanBrightnessError(imgX, imgY):
    #In the start the before and after picture get converted into grey scale picture
    #This is done because of greyscale also can be seen at as brightness levels
    imgXGrey = cv2.cvtColor(imgX, cv2.COLOR_BGR2GRAY)
    imgYGrey = cv2.cvtColor(imgY, cv2.COLOR_BGR2GRAY)

    #After the convertion the formula can be calculated for the 2 pictures by doing as seen down under.
    result = abs(np.mean(imgXGrey)-np.mean(imgYGrey))
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
    return avgGradientResult


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

print(f'Mean Brightness Error: {MeanBrightnessError(imageBefore, imageAfter)}')
print(f'Average Gradient: {AverageGradient(imageBefore, imageAfter)}')
print(f'Patch-based Contrast Quality Index (PCQI): {ContrastQualityIndex(imageBefore, imageAfter, 32)}')
print(f'Underwater colour image quality evaluation metric (UCIQE): {UnderwaterQualityEvaluation(imageBefore, imageAfter)}')
