import os
import sys
os.system('cls')
import subprocess

import numpy as np
import cv2
from matplotlib import pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Dehazing import dehaze as dh
from P3.Objective_testing import Objective_testing as ot

def histogram_equalization(img_in):
    # Segregate color streams
    b, g, r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    
    # Calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)
    
    # Mask all zeros (CDF)
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    
    # Normalize the cdf
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')
    
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')
    
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
    
    # Merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
    
    img_out = cv2.merge((img_b, img_g, img_r))
    
    # Validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))

    return img_out

input_folder = 'Results\Dehaze test set\Dehazed images'
output_folder = 'Results\Dehaze test set\Dehazed images'
type = 'premade' # 'Histogram equalisation', 'dehaze' or 'premade'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if  type == 'Histogram equalisation':
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            equ = histogram_equalization(img)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, equ)
            #ot.OTmethodsSingleImage(img_path,equ)

if type == 'dehaze':
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            dhz = dh.dehaze(img)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, dhz)
            #ot.OTmethodsSingleImage(img_path,dhz)

if type == 'premade':
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            img_path_dehazed = os.path.join('Results\Dehaze test set\Dehazed images', filename)
            img_dehazed = cv2.imread(img_path_dehazed)
            img_path_original = os.path.join('Results\Dehaze test set', filename)
            ot.OTmethodsSingleImage(img_path_dehazed,img_path_original)