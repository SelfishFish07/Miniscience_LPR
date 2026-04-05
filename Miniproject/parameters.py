import numpy as np
from dataclasses import dataclass

@dataclass
class Acousticparameters:
    freq: float = 38e3

    c_medium: float = 343.0
    rho_medium: float = 1.27

    c_trans: float = 6320.0
    rho_trans: float = 2770.0

    c_holo: float = 1054.0
    rho_holo: float = 1080.0
    alpha_holo: float = 5.5

    #Domain parameters
    size_x: float = 48e-3
    size_y: float = 48e-3
    dx: float = 200e-6
    z_target: float = 30e-3

    #Factor for domain expansion
    pad_factor: int = 3

    #Initial thickness
    T0: float = 1e-3

    #Sphere parameters
    rho_sph: float = 2500
    c_l: float = 5600
    c_t: float = 3400

    @property
    def omega(self):
        return 2 * np.pi * self.freq
    
    @property
    def lambda_medium(self):
        return self.c_medium / self.freq
    
    @property
    def lambda_holo(self):
        return self.c_holo / self.freq
    
    @property
    def k_medium(self):
        return self.omega / self.c_medium
    
    @property
    def k_holo(self):
        return self.omega / self.c_holo
    
    @property
    def k_l(self):
        return self.omega / self.c_l
    
    @property
    def k_t(self):
        return self.omega / self.c_t
    
    @property
    def sigma(self):
        return (self.c_l**2 / 2 - self.c_t**2) / (self.c_l**2 - self.c_t**2)

    @property
    def nx(self) -> int:
        return int(np.ceil(self.size_x / self.dx))
    
    @property
    def ny(self) -> int:
        return int(np.ceil(self.size_y / self.dx))
    
    @property
    def Nx(self) -> int:
        return self.nx * self.pad_factor
    
    @property
    def Ny(self) -> int:
        return self.ny * self.pad_factor
    
    @property
    def imp_medium(self):
        return self.rho_medium * self.c_medium

    @property
    def imp_holo(self):
        return self.rho_holo * self.c_holo
    
    @property
    def imp_trans(self):
        return self.rho_trans * self.c_trans