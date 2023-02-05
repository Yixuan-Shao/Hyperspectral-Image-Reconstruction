# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
Functions for evaluating the model
"""

import numpy as np

# Evaluate the spectral RMSRE (root mean squared relative error) from the 
# spectrum of reconstructed hyperspectral image and ground truth
def spectrumRMSRE(reconstructed_hsi, groundTruth_hsi, y_range, x_range):
    if (type(y_range)==int):
        y_range = slice(y_range, y_range+1)
    else:
        y_range = slice(y_range[0], y_range[1])
    if (type(x_range)==int):
        x_range = slice(x_range, x_range+1)
    else:
        x_range = slice(x_range[0], x_range[1])
        
    # 3:26 correspond to the 430 to 650 nm wavelength range
    reconstructed_patch = reconstructed_hsi[y_range, x_range, 3:26]
    groundTruth_patch = groundTruth_hsi[y_range, x_range, 3:26]
    reconstructed_spectrum = np.mean(reconstructed_patch, axis=(0,1))
    groundTruth_spectrum = np.mean(groundTruth_patch, axis=(0,1))
    relative_error = (reconstructed_spectrum - groundTruth_spectrum)/(groundTruth_spectrum+1e-8)
    RMSRE = np.sqrt((relative_error ** 2).mean())
    
    return RMSRE, reconstructed_spectrum, groundTruth_spectrum