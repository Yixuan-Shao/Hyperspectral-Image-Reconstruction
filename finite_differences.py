# Operators for finding spatial gradient, spectral gradient and their conjugate
# operators.
import numpy as np

def opDx_(x, axis):
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(0, 1)
    Dx = np.diff(x, axis=axis, append=x[tuple(slc)])
    return Dx

def opDtx_(diffx, axis):
    Dtx = -diffx + np.roll(diffx, 1, axis=axis)
    return Dtx

def opDx(x):
    Dx = opDx_(x, axis=1)
    Dy = opDx_(x, axis=0)
    if len(Dx.shape) <= 3:
        return np.stack((Dx, Dy), axis=-4)
    else:
        print("dimension wrong!")
        # return np.concatenate((Dx, Dy), axis=-3)

def opDtx(x):
    Dtx = opDtx_(x[0, ...], axis=1)
    Dty = opDtx_(x[1, ...], axis=0)
    return Dtx + Dty

def opDlda(x):
    axis = -1
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(0,1)
    Dx = np.diff(x, axis=axis, append=x[tuple(slc)])
    return Dx

def opDtlda(diffx):
    axis = -1
    Dtx = -diffx + np.roll(diffx, 1, axis=axis)
    return Dtx