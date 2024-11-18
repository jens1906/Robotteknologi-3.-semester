def DirectionalPatchBasedContrastQualityIndex(imgB, imgA, PatchSize, C=1e-5):
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
            
            # Directional PCQI calculation for the patch
            # Introducing directionality by adding (contrastB - contrastA) term
            pcqi_patch = (2 * contrastB * contrastA + C) / (contrastB**2 + contrastA**2 + C) * (contrastB - contrastA)
            
            # Accumulate directional PCQI score for all patches
            Total_PCQI += pcqi_patch
            num_patches += 1

    # Average the directional PCQI over all patches
    avg_PCQI = Total_PCQI / num_patches
    return avg_PCQI




def CompareContrastQualityIndex(imgB, imgA, PatchSize):
    Total_PCQI = 0
    for y in range(0, imgB.shape[0], PatchSize):
        for x in range(0, imgB.shape[1], PatchSize):
            imagePatches = [imgB[y:y+PatchSize, x:x+PatchSize],
                            imgA[y:y+PatchSize, x:x+PatchSize]]
            imgPatchLuminences = [np.mean(imagePatches[0]),
                                  np.mean(imagePatches[1])]
            #luminanceDifference = imgPatchLuminences[0] - imgPatchLuminences[1]
            luminanceComparison = (imgPatchLuminences[0] - imgPatchLuminences[1]) / 2
            Total_PCQI += luminanceComparison * imgPatchLuminences[0] * imgPatchLuminences[1]
    Patches = (imgB.shape[0] // PatchSize) * (imgB.shape[1] // PatchSize)
    avgPatchContrastResult = Total_PCQI / Patches 
    if avgPatchContrastResult>0:
        result = "Worse"
    elif avgPatchContrastResult<0:
        result = "Better"
    else:
        result = "No change"

    return result