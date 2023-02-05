# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
Function for Step2: Recover the gradient of the hyperspectral image with 
respect to dispersion direction x 
Using ADMM method
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from tqdm import tqdm

# These functions implements the finite differences method
from finite_differences import *
# Functions for the forward model and their conjugate functions
from forward import *
# Convert edge pixels into a vector and back
from edgeOperator import Mb, Mf

def step2(j, edge, E, h, w, c, alpha2=1e-3, beta2=1e-3, rho3=1e-2, num_iters=10, 
          cg_iters=10, cg_tolerance=1e-12):
    vx = np.ones((E, c)) # Gradient of the hyperspectral image's edge pixels
    z3 = np.zeros((E, c)) # ADMM variable
    u3 = np.zeros((E, c)) # ADMM variable
    
    # Vectorize the input and back
    vx2vec = lambda vx: np.reshape(vx, (vx.size, 1))
    vec2vx = lambda x: np.reshape(x, (E, c))
    
    # Coefficient matrix for the conjugate gradient method
    def AtA2(vx):
        vx = vec2vx(vx)
        term1 = Mf(forwardT(forward(Mb(vx, edge, (h, w, c)))), edge)
        term2 = beta2 * Mf(opDtx_(opDx_(Mb(vx, edge, (h, w, c)), axis=1), axis=1), edge)
        term3 = rho3/2 * opDtlda(opDlda(vx))
        out = term1 + term2 + term3
        return vx2vec(out)
    
    # ADMM
    for it in tqdm(range(num_iters)):
    
        # x update using cg solver
        v3 = z3-u3  # ADMM variable
        
        opAtA2 = LinearOperator((vx.size, vx.size), matvec = AtA2)
        term1 = Mf(forwardT(opDx_(j, axis=1)), edge)
        term2 = rho3/2 * opDtlda(v3)
        Atb2 = vx2vec(term1 + term2)
        vx_vec, _ = cg(opAtA2, Atb2, tol=cg_tolerance, maxiter=cg_iters)
        vx = vec2vx(vx_vec)
    
        # z update - soft shrinkage    
        kappa3 = alpha2 / rho3
        v3 = opDlda(vx) + u3
        
        eps = np.finfo(np.float64).eps
        z3 = np.maximum(1 - kappa3/(np.abs(v3)+eps), 0) * v3
        
        # u-update
        u3 = u3 + opDlda(vx) - z3
        
    return vx