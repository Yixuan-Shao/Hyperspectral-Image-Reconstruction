# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
This script reconstructs a hyperspectral image from a dispersed image
using the example of the real and fake lemon slices
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio 

# These functions implements the finite differences method
from finite_differences import *
# Functions for the forward model and their conjugate functions
from forward import *
# Detect edge pixels, convert edge pixels into a vector and back
from edgeOperator import edgeDetect, Mb, Mf
# Functions for evaluating the accuracy of the model
from evaluate_model import spectrumRMSRE

# 3 steps in this hyperspectral image reconstruction algorithm
from step1 import step1
from step2 import step2
from step3 import step3

# Read the ground truth hyperspectral image data
name = 'fake_and_real_lemon_slices_ms/fake_and_real_lemon_slices_ms_'
raw = np.zeros((512, 512, 31))
for i in range(31):
    if i<9:
        s = "0"+str(i+1)
    else:
        s = str(i+1)
    
    img = io.imread(f'{name}{s}.png').astype(float) / 255 / 255 * 8
    raw[...,i] = img
    
raw = raw[256:, 128:384, :]
raw[:, -32:, :] = 0 # Due to dispersion, some of the pixels' spectral information is lost
h, w, c = raw.shape # Height, width, and channels

# Forward model. 
# Generate the dispersed image, which is captured by the camera
j = forward(raw)
# Add 10% Gaussian noise
j += np.random.randn(h, w, 3) * np.mean(j) * 0.1


# Step1: Align j into a non-dispersed image
alpha1 = 1e-3
beta1 = 1e-3
rho1 = 1e-2
rho2 = 1e-2
num_iters = 10
cg_iters = 10           # number of iterations for CG solver
cg_tolerance = 1e-12    # convergence tolerance of cg solver
aligned_hsi = step1(j, h, w, c, alpha1=alpha1, beta1=beta1, rho1=rho1, rho2=rho2,
                    num_iters=num_iters, cg_iters=cg_iters, cg_tolerance=cg_tolerance)

# Convert aligned hyperspectral image to RGB image
aligned_img = hsi2rgb(aligned_hsi)
# Obtain the edge of the aligned image
edge = edgeDetect(aligned_img, ksize=3, sigma=0.5, percentile=92)
E = Mf(aligned_hsi, edge).shape[0] # Number of pixels in the edge

# Step2: Recover the gradient of the hyperspectral image with 
# respect to dispersion direction x 
alpha2 = 1e-3
beta2 = 1e-3
rho3 = 1e-2
num_iters = 10
cg_iters = 10           # number of iterations for CG solver
cg_tolerance = 1e-12    # convergence tolerance of cg solver
# Find the gradient of the hyperspectral image's edge pixels
vx = step2(j, edge, E, h, w, c, alpha2=alpha2, beta2=beta2, rho3=rho3,
           num_iters=num_iters, cg_iters=cg_iters, cg_tolerance=cg_tolerance)
# The gradient of the hyperspectral image in x direction
gx = Mb(vx, edge, (h, w, c))


# Step3: Recover hyperspectral image
alpha3 = 2e-2
beta3 = 1e-3
cg_iters = 100           # number of iterations for CG solver
cg_tolerance = 1e-12    # convergence tolerance of cg solver
reconstructed_hsi = step3(j, gx, h, w, c, alpha3=alpha3, beta3=beta3, 
                          cg_iters=cg_iters, cg_tolerance=cg_tolerance)

# Clip the margin
raw = raw[:, :-32, :]
reconstructed_hsi = reconstructed_hsi[:, :-32, :]

# Convert reconstructed hyperspectral image to RGB image
recovered_img = hsi2rgb(reconstructed_hsi)
recovered_img = np.clip(recovered_img, 0, 1)

# Get ground truth RGB image from ground truth hyperspectral image
nondispersed_img = hsi2rgb(raw)



# Results
plt.figure()
plt.imshow(nondispersed_img)
plt.axis('off')
plt.savefig("result_lemon/groundtruthRGBimage.png", bbox_inches='tight')
plt.savefig("result_lemon/groundtruthRGBimage.svg", dpi=300, format='svg',
            bbox_inches='tight')

plt.figure()
plt.imshow(recovered_img)
plt.axis('off')
plt.savefig("result_lemon/recoveredRGBimage.png", bbox_inches='tight')
plt.savefig("result_lemon/recoveredRGBimage.svg", dpi=300, format='svg',
            bbox_inches='tight')

plt.figure()
plt.imshow(j)
plt.axis('off')
plt.savefig("result_lemon/capturedRGBimage.png", bbox_inches='tight')
plt.savefig("result_lemon/capturedRGBimage.svg", dpi=300, format='svg',
            bbox_inches='tight')


plt.figure()
plt.imshow(edge[:, :-32], cmap ='gray')
plt.axis('off')
plt.savefig("result_lemon/edge.png", bbox_inches='tight')
plt.savefig("result_lemon/edge.svg", dpi=300, format='svg',
            bbox_inches='tight')



# Spectral result
fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(11, 8))
for i in range(23):
    wavelength = 430 + 10 * i
    single_wavelength_hsi = np.zeros(reconstructed_hsi.shape)
    single_wavelength_hsi[...,i+3] = reconstructed_hsi[...,i+3] * 15
    single_wavelength_rgbimg = np.clip(hsi2rgb(single_wavelength_hsi), 0, 1)
    
    row, col = i//6, i%6
    axs[row, col].imshow(single_wavelength_rgbimg)
    axs[row, col].axis('off')
    axs[row, col].text(70, 30, str(wavelength)+"nm", color='white')
plt.delaxes(axs[3, 5])
plt.subplots_adjust(wspace=0.01,hspace=0.02)
plt.savefig("result_lemon/hsi.png", bbox_inches='tight')
plt.savefig("result_lemon/hsi.svg", dpi=300, format='svg',
            bbox_inches='tight')

# Real lemon
y_range = (133, 138)
x_range = (186, 190)
RMSRE, reconstructed_spectrum_real, groundTruth_spectrum_real = spectrumRMSRE(
                    reconstructed_hsi, raw, y_range=y_range, x_range=x_range)
reconstructed_spectrum_real = reconstructed_spectrum_real / 1.8
groundTruth_spectrum_real = groundTruth_spectrum_real / 1.8

# Fake lemon
y_range = (140, 145)
x_range = (85, 90)
RMSRE, reconstructed_spectrum_fake, groundTruth_spectrum_fake = spectrumRMSRE(
                    reconstructed_hsi, raw, y_range=y_range, x_range=x_range)


plt.figure(figsize=(6,5))
plt.plot(np.arange(430, 660, 10), groundTruth_spectrum_real, 
         label="Ground truth of the real lemon", linewidth=3)
plt.plot(np.arange(430, 660, 10), reconstructed_spectrum_real, 'r.',
         label="Recovered spectrum of the real lemon", markersize=12)
plt.plot(np.arange(430, 660, 10), reconstructed_spectrum_real, 'r-', linewidth=3)
plt.plot(np.arange(430, 660, 10), groundTruth_spectrum_fake, 
         label="Ground truth of the fake lemon", linewidth=3)
plt.plot(np.arange(430, 660, 10), reconstructed_spectrum_fake, 'b.', 
         label="Recovered spectrum of the fake lemon", markersize=12)
plt.plot(np.arange(430, 660, 10), reconstructed_spectrum_fake, 'b-', linewidth=3)
plt.legend()
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Radiance", fontsize=14)
plt.ylim(0)
plt.savefig("result_lemon/spectrum.png", bbox_inches='tight')
plt.savefig("result_lemon/spectrum.svg", dpi=300, format='svg',
            bbox_inches='tight')


# Mark the patches where the spectra are from
plt.figure()
plt.imshow(recovered_img)
plt.axis('off')
ax = plt.gca()
rect = patches.Rectangle((186,133), 5, 5, fill=False, edgecolor = 'red',linewidth=2)
ax.add_patch(rect)
rect = patches.Rectangle((85,140), 5, 5, fill=False, edgecolor = 'red',linewidth=2)
ax.add_patch(rect)
plt.savefig("result_lemon/mark.png", bbox_inches='tight')
plt.savefig("result_lemon/mark.svg", dpi=300, format='svg',
            bbox_inches='tight')



PSNR = round(peak_signal_noise_ratio(recovered_img, nondispersed_img),2)
print(PSNR) # 26.76

