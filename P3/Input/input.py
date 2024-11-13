import os
os.system('cls')
import cv2
from vmbpy import *
import matplotlib.pyplot as plt
global InputTest
InputTest = False


def get_image():
    try:
        with VmbSystem.get_instance() as vmb:
            cams = vmb.get_all_cameras()
            if len(cams) == 0:
                print('No cameras found')
                exit(1)
            with cams[0] as cam:
                frame = cam.get_frame()
                bayer_image = frame.as_numpy_ndarray()
                rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2RGB)
                if InputTest:
                    print(rgb_image.shape)
                    plt.imshow(rgb_image)
                    plt.show()
        return rgb_image
    except Exception as e:
        print(e)
        exit(1)
if InputTest:
    get_image()

