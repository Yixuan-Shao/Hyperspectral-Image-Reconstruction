# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
Functions for finding edges and manipulating edges.
"""

import cv2
import numpy as np
from finite_differences import opDx_

# A binary filter operator that provides a reliable estimate of edges along the x-axis.
 # First find pixels where the spatial gradient along the x-axis is 
 # larger than a certain threshold. Then blur the obtained image using a 
 # Gaussian low-pass filter. Finally apply a binary filter again to 
 # find pixels where the spatial gradient along the x-axis is larger than 
 # the threshold.
def edgeDetect(recovered_img, ksize=5, sigma=0.5, percentile=90):
    gx = opDx_(recovered_img, axis=1)
    gx = np.absolute(gx)
    threshold = np.percentile(gx, percentile)
    edge = np.array(gx>=threshold, dtype=np.float64)
    
    ksize = (ksize, ksize)
    edgeblur = cv2.GaussianBlur(edge, ksize=ksize, sigmaX=sigma)
    result = np.array(edgeblur>=threshold, dtype=np.float64)
    edge = np.max(result, 2)
    
    return edge


# Extracts the values at edge pixels from the full size image.
def Mf(gx, edge):
    nonzero = np.nonzero(edge)
    vx = [gx[..., c_ind][nonzero] for c_ind in range(gx.shape[-1])]
    vx = np.array(vx).T
    return vx


# Interpolates edge pixel values with zeros to restore the full size image.
def Mb(vx, edge, shape):
    gx = np.zeros(shape)
    nonzero = np.nonzero(edge)
    for c_ind in range(vx.shape[-1]):
        gx[..., c_ind][nonzero] = vx[:, c_ind]
    return gx