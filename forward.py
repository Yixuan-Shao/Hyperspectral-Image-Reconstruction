# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
Functions for the dispersive image formation model
"""

import numpy as np
from numpy.fft import fft2, ifft2

import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio 
import skimage.color

import matplotlib.pyplot as plt

from scipy.io import loadmat

# Convert hyperspectrum to XYZ color space based on CIE 1931
xyzbar = loadmat('xyzbar.mat')
xyzbar = xyzbar['xyzbar']
xyzbar = xyzbar[0:31:1, :]
xyzbar = xyzbar / np.sum(xyzbar, axis=0)

# Conversion matrix between XYZ and RGB color space
xyz2rgb = np.array([[0.41847, -0.15866, -0.082835],
                    [-0.091169, 0.25243, 0.015708],
                    [0.0009209, -0.0025498, 0.1786]])

def disperse(hsi):
    disperse_hsi = np.zeros(hsi.shape)
    for i in range(hsi.shape[-1]):
        disperse_hsi[..., i] = np.roll(hsi[...,i], i, axis=1)
    return disperse_hsi

def disperseT(hsi):
    disperse_hsi = np.zeros(hsi.shape)
    for i in range(hsi.shape[-1]):
        disperse_hsi[..., i] = np.roll(hsi[...,i], -i, axis=1)
    return disperse_hsi

def hsi2rgb(hsi, xyzbar=xyzbar):
    xyzimg = np.sum(hsi[..., None] * xyzbar, axis=2)
    rgbimg = np.sum(xyzimg[:, :, None, :] * xyz2rgb, axis=3)
    return rgbimg

def hsi2rgbT(rgbimg, xyzbar=xyzbar):
    xyzimg = np.sum(rgbimg[:, :, None, :] * xyz2rgb.T, axis=3)
    hsi = np.sum(xyzimg[..., None] * xyzbar.T, axis=2)
    return hsi

def forward(hsi):
    return hsi2rgb(disperse(hsi))

def forwardT(rgbimg):
    return disperseT(hsi2rgbT(rgbimg))


if __name__ == '__main__':
    name = 'stuffed_toys_ms/stuffed_toys_ms_'
    raw = np.zeros((512, 512, 31))
    for i in range(31):
        if i<9:
            s = "0"+str(i+1)
        else:
            s = str(i+1)
        
        img = io.imread(f'{name}{s}.png').astype(float) / 255 / 255 * 8
        raw[...,i] = img
        
            
    raw = raw[340:468, 128:256, :]
    raw[:, -31:, :] = 0
    # plt.figure()
    # plt.imshow(raw[...,5])
    
    nondispersed_img = hsi2rgb(raw)
    dispersed_img = forward(raw)
    plt.figure()
    plt.subplot(121)
    plt.imshow(nondispersed_img)
    plt.subplot(122)
    plt.imshow(dispersed_img)