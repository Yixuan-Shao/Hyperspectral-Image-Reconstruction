# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
Function for Step3: Recover the final hyperspectral image
Using conjugate gradient method
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

# These functions implements the finite differences method
from finite_differences import *
# Functions for the forward model and their conjugate functions
from forward import *

def step3(j, gx, h, w, c, alpha3=2e-2, beta3=5e-4, cg_iters=300, cg_tolerance=1e-12):
    
    # Vectorize the input and inverse-vectorize it back to a hyperspectral image
    hsi2vec = lambda x: np.reshape(x, (x.size, 1))
    vec2hsi = lambda x: np.reshape(x, (h, w, c))
    
    # Coefficient matrix for the conjugate gradient method
    def AtA3(x):
        x = vec2hsi(x)
        term1 = forwardT(forward(x))
        term2 = alpha3 * opDtx_(opDx_(x, axis=1), axis=1)
        # term3 = beta3 * opDtlda(opDtlda(opDlda(opDlda(x))))
        term3 = beta3 * opDtlda(opDlda(x))
        out = term1 + term2 + term3
        return hsi2vec(out)
    
    opAtA3 = LinearOperator((h*w*c, h*w*c), matvec = AtA3)
    term1 = forwardT(j)
    term2 = alpha3 * opDtx(gx)
    Atb3 = hsi2vec(term1 + term2)
    x_vec, _ = cg(opAtA3, Atb3, tol=cg_tolerance, maxiter=cg_iters)
    x = vec2hsi(x_vec)
    
    return x
