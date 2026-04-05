import numpy as np
from parameters import Acousticparameters

# Reference: Melde, K., et al. "Holograms for acoustics." Nature (2016)

class ASMPropagator:
    def __init__(self, params, Kx, Ky, z):
        self.z = z
        k = params.k_medium

        #Elimination of evanescent waves
        propagation_mask  = (Kx**2 + Ky**2) <= k**2
        D = params.size_x
        #Eq. (6)
        angular_mask = np.sqrt(Kx**2 + Ky**2) <= k * D / np.sqrt(D**2 + self.z**2)

        mask = propagation_mask & angular_mask

        kz = np.sqrt(np.where(mask,k**2 - (Kx**2 + Ky**2),0))

        #Eq. (3)
        self.H = np.exp(1j * kz * z) * mask
    
    def forward(self, field_z0):
        #Eq. (3-4)
        spectrum_0 = np.fft.fft2(field_z0)
        spectrum_z = spectrum_0 * self.H
        field_z = np.fft.ifft2(spectrum_z)
        return field_z
    
    def backward(self, field_z):
        #Eq. (5)
        spectrum_z = np.fft.fft2(field_z)
        spectrum_0 = spectrum_z * np.conjugate(self.H)
        field_z0 = np.fft.ifft2(spectrum_0)
        return field_z0