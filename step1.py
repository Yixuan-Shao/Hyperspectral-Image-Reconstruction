# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
Function for Step1: Align the captured dispersed image into a non-dispersed image.
Using ADMM method
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from tqdm import tqdm

# These functions implements the finite differences method
from finite_differences import *
# Functions for the forward model and their conjugate functions
from forward import *

def step1(j, h, w, c, alpha1=1e-3, beta1=1e-3, rho1=1e-2, rho2=1e-2, num_iters=10, 
          cg_iters=10, cg_tolerance=1e-12):
    
    x = np.zeros((h, w, c)) # Aligned hyperspectral image
    z1 = np.zeros((2, h, w, c)) # ADMM variable
    z2 = np.zeros((2, h, w, c)) # ADMM variable
    u1 = np.zeros((2, h, w, c)) # ADMM variable
    u2 = np.zeros((2, h, w, c)) # ADMM variable
    
    # Vectorize the input and inverse-vectorize it back to a hyperspectral image
    hsi2vec = lambda x: np.reshape(x, (x.size, 1))
    vec2hsi = lambda x: np.reshape(x, (h, w, c))
    
    # Coefficient matrix for the conjugate gradient method
    def AtA(x):
        x = vec2hsi(x)
        term1 = forwardT(forward(x))
        term2 = rho1/2 * opDtx(opDx(x))
        term3 = rho2/2 * opDtx(opDtlda(opDlda(opDx(x))))
        out = term1 + term2 + term3
        return hsi2vec(out)
    
    # ADMM
    for it in tqdm(range(num_iters)):
        # x update using cg solver
        v1 = z1-u1 # ADMM variable
        v2 = z2-u2 # ADMM variable
        opAtA = LinearOperator((x.size, x.size), matvec = AtA)
        term1 = forwardT(j)
        term2 = rho1/2 * opDtx(v1)
        term3 = rho2/2 * opDtx(opDtlda(v2))
        Atb = hsi2vec(term1 + term2 + term3)
        x_vec, _ = cg(opAtA, Atb, tol=cg_tolerance, maxiter=cg_iters)
        x = vec2hsi(x_vec)
    
        # z update - soft shrinkage    
        kappa1 = alpha1 / rho1
        kappa2 = beta1 / rho2
        v1 = opDx(x) + u1 # ADMM variable
        v2 = opDlda(opDx(x)) + u2 # ADMM variable
        
        eps = np.finfo(np.float64).eps
        z1 = np.maximum(1 - kappa1/(np.abs(v1)+eps), 0) * v1
        z2 = np.maximum(1 - kappa2/(np.abs(v2)+eps), 0) * v2
        
        # u-update
        u1 = u1 + opDx(x) - z1
        u2 = u2 + opDx(x) - z2
        
    return x