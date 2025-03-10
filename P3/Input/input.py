import cv2
from vmbpy import *
import matplotlib.pyplot as plt

import time

def initialize_camera():
    try:
        
        with VmbSystem.get_instance() as vmb: #This line takes approx 0.7s
            cams = vmb.get_all_cameras()
            if len(cams) == 0:
                print('No cameras found')
                exit(1)
            with cams[0] as cam: # This line takes approx 1.0s
                cam.ExposureAuto.set('Off') # Set exposure to auto
                cam.GainAuto.set('Off') # Set gain to off
                cam.Gain.set(0) # Set gain to 0 
                return cam 
    except Exception as e:
        print(e)
        exit(1)

def get_image(exposure):
    try:
        with VmbSystem.get_instance() as vmb: #This line takes approx 0.7s
            cams = vmb.get_all_cameras()
            if len(cams) == 0:
                print('No cameras found')
                exit(1)
            with cams[0] as cam: # This line takes approx 1.0s
                cam.ExposureTimeAbs.set(exposure)
                frame = cam.get_frame() 
                bayer_image = frame.as_numpy_ndarray()
                rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2RGB)   
        return rgb_image
    except Exception as e:
        print(e)
        exit(1)

def get_exposure():
    try:
        with VmbSystem.get_instance() as vmb: #This line takes approx 0.7s
            cams = vmb.get_all_cameras()
            if len(cams) == 0:
                print('No cameras found')
                exit(1)
            with cams[0] as cam: # This line takes approx 1.0s
                return cam.ExposureTimeAbs.get()
    except Exception as e:
        print(e)
        exit(1)

