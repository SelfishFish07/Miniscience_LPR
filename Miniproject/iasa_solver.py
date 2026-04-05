import numpy as np
from parameters import Acousticparameters
from propagator import ASMPropagator

# Reference: Melde, K., et al. "Holograms for acoustics." Nature (2016)


def calculate_thickness(phase, params):
    #Eq. (8)
    thickness = phase / (params.omega * (1 / params.c_holo - 1 / params.c_medium))
    return params.T0 + thickness - np.min(thickness)

def calculate_transmission(phase, params):
    Zm = params.imp_medium
    Zh = params.imp_holo
    Zt = params.imp_trans
    km = params.k_medium
    kh = params.k_holo

    T = calculate_thickness(phase, params)

    #Eq. (7)
    alpha_t = 4 * Zt * Zh**2 * Zm / ((Zh * (Zt + Zm) * np.cos(kh * T))**2 + ((Zh**2 + Zt * Zm) * np.sin(kh * T))**2)
    # return alpha_t
    return np.ones_like(T)

def run_iasa(target_amplitude, p0_amplitude, propagator, params, iterations = 50):
    field_hologram = p0_amplitude * np.exp(1j * 0.0)

    for i in range(iterations):
        field_target = propagator.forward(field_hologram)

        phase_target = np.angle(field_target)
        field_target_modified = target_amplitude * np.exp(1j * phase_target)

        field_holo_back = propagator.backward(field_target_modified)

        phase_retrieved = np.angle(field_holo_back)

        alpha_t = calculate_transmission(phase_retrieved, params)
        field_hologram = np.sqrt(alpha_t) * p0_amplitude * np.exp(1j * phase_retrieved)

        if (i % 10 == 0):
            print(f"{i} итерация завершена")
        
    return phase_retrieved
