import numpy as np
import numpy.fft as ft

def fft2(x):
    x = ft.ifftshift(x, axes=(-2, -1))
    X = ft.fft2(x, norm='ortho')
    X = ft.fftshift(X, axes=(-2, -1))
    return X

def ifft2(X):
    X = ft.ifftshift(X, axes=(-2, -1))
    x = ft.ifft2(X, norm='ortho')
    x = ft.fftshift(x, axes=(-2, -1))
    return x

def root_sum_of_squares(x):
    x_abs = np.abs(x)
    x_rsos = np.sqrt(np.sum(x_abs**2, axis=0))
    return x_rsos


