import rawpy
import cv2 as cv
import numpy as np
import os

def convert_raw_to_bgr(image_path):
    with rawpy.imread(image_path) as raw:
        rgb_image = raw.postprocess(
            use_auto_wb=True,
            no_auto_bright=False,
            output_bps=16,
            highlight_mode=2,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.PPG #Used is PPG due to spped, maybe use AHD for quality or LINEAR
        )
    bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
    return bgr_image

def resize_image(image, width=800):
    height = int(image.shape[0] * (width / image.shape[1]))
    resized_image = cv.resize(image, (width, height))
    return resized_image

def show_image(image):
    resized_image = resize_image(image)
    cv.imshow('Converted Image', resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Im stupido pls dont hate my file path
#script_dir = os.path.dirname(__file__)
#image_path = os.path.join(script_dir, 'hazy15.dng') 

#converted_image = convert_raw_to_bgr(image_path)
#show_image(converted_image)