import numpy as np
from parameters import Acousticparameters


def generate_grids(params):

    Nx = params.Nx
    Ny = params.Ny
    dx = params.dx

    x_vec = np.arange(-Nx//2, Nx//2) * dx
    y_vec = np.arange(-Ny//2, Ny//2) * dx

    kx_vec = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky_vec = 2 * np.pi * np.fft.fftfreq(Ny, dx)

    X, Y = np.meshgrid(x_vec, y_vec)
    Kx, Ky = np.meshgrid(kx_vec, ky_vec)

    return X, Y, Kx, Ky

def create_source_amplitude(X, Y, radius, inner_radius = 0e-3):
    mask1 = (X**2 + Y**2) <= radius**2
    mask2 = (X**2 + Y**2) >= inner_radius**2
    mask = mask1 & mask2
    return np.where(mask, 1.0, 0.0)

