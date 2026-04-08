import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from parameters import Acousticparameters
from propagator import  ASMPropagator
from domain import generate_grids, create_source_amplitude
from scipy.special import spherical_jn, spherical_yn, sph_harm_y
from scipy.interpolate import RegularGridInterpolator

# Reference: Sapozhnikov, O., et al. "Radiation force of an arbitrary acoustic beam on an elastic sphere in a fluid.", J. Acoust. Soc. Am. (2013)

def calculate_cn_array(n_max: int, a, params: Acousticparameters):
    n = np.arange(n_max + 2)
    
    rho_medium = params.rho_medium
    rho_sp = params.rho_sph
    sigma = params.sigma
    k_t = params.k_t
    k_l = params.k_l
    k = params.k_medium

    z_t = k_t * a
    z_l = k_l * a
    z_k = k * a

    #Spherical Bessel functions and their derivatives
    j_t = spherical_jn(n, z_t)
    dj_t = spherical_jn(n, z_t, derivative=True)
    j_l = spherical_jn(n, z_l)
    dj_l = spherical_jn(n, z_l, derivative=True)
    j_k = spherical_jn(n, z_k)
    dj_k = spherical_jn(n, z_k, derivative=True)
    y_k = spherical_yn(n, z_k)
    dy_k = spherical_yn(n, z_k, derivative=True)

    ddj_t = (n * (n + 1) / z_t**2 - 1) * j_t - (2 / z_t) * dj_t
    ddj_l = (n * (n + 1) / z_l**2 - 1) * j_l - (2 / z_l) * dj_l

    #Below Eq. (13)
    alpha_n = j_l - z_l * dj_l
    beta_n = (n**2 + n - 2) * j_t + z_t**2 * ddj_t
    hi_n = z_l * dj_l
    delta_n = 2 * n * (n + 1) * j_t
    epsilon_n = z_l**2 * (j_l * sigma / (1 - 2 * sigma) - ddj_l)
    nu_n = 2 * n * (n + 1) * (j_t - z_t * dj_t)

    #Eq. (13)
    G_n = (rho_medium * z_t**2 / (2 * rho_sp)) * ((alpha_n * delta_n + hi_n * beta_n) / (alpha_n * nu_n + epsilon_n * beta_n))

    #Spherical Hankel functions of the ﬁrst kind
    h_n = j_k + 1j * y_k
    d_h_n = dj_k + 1j * dy_k

    #Eq. (12)
    c_n = - (G_n * j_k - z_k * dj_k) / (G_n * h_n - z_k * d_h_n)
    
    return c_n

def calculate_psi_n_array(c_n):
    #Eq. (49)
    return (1 + 2 * c_n[:-1]) * (1 + 2 * np.conjugate(c_n[1:])) - 1

def calculate_Hnm(field_z, params: Acousticparameters, Kx, Ky, n_max: int):
    k = params.k_medium

    mask =  Kx**2 + Ky**2 <= k**2
    Kz = np.sqrt(np.where(mask, k**2 - (Kx**2 + Ky**2 ), 0))
    theta_k = np.arccos(Kz / k)
    phi_k = np.arctan2(Ky, Kx)


    spectrum = np.fft.fft2(field_z)
    H_dict = {}

    for n in range(n_max + 2):
        for m in range(-n, n+1):
            #Spherical harmonics
            Ynm = sph_harm_y(n, m, theta_k, phi_k)
            #Eq. (34)
            H_dict[(n, m)] = (2 * np.pi)**2 * np.fft.ifft2(spectrum * np.conjugate(Ynm) * mask)
    return H_dict

def calculate_forces(a, field_z, Kx, Ky, params: Acousticparameters, n_max: int, cropped: bool = False, **kwargs):
    c_n = calculate_cn_array(n_max, a, params)
    psi_n = calculate_psi_n_array(c_n)
    H_dict = calculate_Hnm(field_z, params, Kx, Ky, n_max)

    #For calculating in cropped region
    if cropped:
        sl1 = kwargs.get('sl1')
        sl2 = kwargs.get('sl2')
        H_dict= {key: val[sl1][sl2] for key, val in H_dict.items()}

    
    rho = params.rho_medium
    c = params.c_medium
    k = params.k_medium

    shape = H_dict[(0,0)].shape

    #Eq. (46-48)
    sum_x = np.zeros(shape, dtype=np.complex128)
    sum_y = np.zeros(shape, dtype=np.complex128)
    sum_z = np.zeros(shape, dtype=np.complex128)

    def get_Hnm(n, m):
        return H_dict.get((n, m), 0)

    for n in range(n_max + 1):

        inner_sum_x = np.zeros(shape, dtype=np.complex128)
        inner_sum_y = np.zeros(shape, dtype=np.complex128)
        inner_sum_z = np.zeros(shape, dtype=np.complex128)

        for m in range(-n, n + 1):

            #Eq. (50)
            A_nm = np.sqrt((n + m + 1) * (n + m + 2) / ((2 * n + 1) * (2 * n + 3)))
            #Eq. (51)
            B_nm = np.sqrt((n + m + 1) * (n - m + 1) / ((2 * n + 1) * (2 * n + 3)))

            inner_sum_x += A_nm * (get_Hnm(n, m) * np.conjugate(get_Hnm(n + 1, m + 1)))
            inner_sum_x -= A_nm * (get_Hnm(n, -m) * np.conjugate(get_Hnm(n + 1, -m - 1)))
            inner_sum_y += A_nm * (get_Hnm(n, m) * np.conjugate(get_Hnm(n + 1, m + 1)))
            inner_sum_y += A_nm * (get_Hnm(n, -m) * np.conjugate(get_Hnm(n + 1, -m - 1)))
            inner_sum_z += B_nm * (get_Hnm(n, m) * np.conjugate(get_Hnm(n + 1, m)))
        sum_x += psi_n[n] * inner_sum_x
        sum_y += psi_n[n] * inner_sum_y
        sum_z += psi_n[n] * inner_sum_z

    coef =  1 / (8 * np.pi**2 * rho * c**2 * k**2) 
    F_x = coef * np.real(sum_x)
    F_y = coef * np.imag(sum_y)
    F_z = -2 * coef * np.real(sum_z)

    return F_x, F_y, F_z

def get_nmax(k, a):
    #Wiscombe criterion
    z = k * a
    n_max = int(np.ceil(z + 4 * z ** (1/3)+ 1))
    return a

def force_vs_radius(rads: np.ndarray, X, Y, field_z, Kx, Ky, params):
    k = params.k_medium
    start_x = (params.Nx - params.nx) // 2
    start_y = (params.Ny - params.ny) // 2
    sl1 = (slice(start_y, start_y + params.ny), slice(start_x, start_x + params.nx))

    h1, h2 = 260, 330
    v1, v2 = 120, 190
    sl2 = (slice(h1, h2), slice(v1, v2))

    Y *= -1
    X_crop = X[sl1][sl2] * 1e3
    Y_crop = Y[sl1][sl2] * 1e3
    x_clice = X_crop[0, :]
    Amp_crop = np.abs(field_z)[sl1][sl2]



    linestyles=['-', '--', '-.', ':']
    colors = ['r', 'b', 'g']

    flag = False
    for i in range(len(rads)):
        n_max = get_nmax(k, rads[i])
        Fx, Fy, _ = calculate_forces(rads[i], field_z, Kx, Ky, params, n_max, cropped=True, sl1 = sl1, sl2 =sl2)
        Fy *= -1
        if flag:
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(Amp_crop, cmap='jet', origin='upper', extent=[X_crop.min(), X_crop.max(),Y_crop.min(), Y_crop.max()])
            fig.colorbar(im, ax=ax, shrink = 0.8)
            step = 4
            factor = np.max([np.max(np.abs(Fx)), np.max(np.abs(Fy))])
            ax.quiver(X_crop[::step, ::step], 
                    Y_crop[::step, ::step], 
                    Fx[::step, ::step] / factor, 
                    Fy[::step, ::step] / factor, 
                    color='white', width=0.003, scale=10)
            
            ax.hlines(Y_crop[15,0], X_crop.min(), X_crop.max(), linestyles='dashed', color = 'r', linewidth = 3)
            ax.set_xlabel("x, mm")
            ax.set_ylabel("y, mm")
            plt.title(f"Forces in x-y direction (a = {int(rads[i]*1e6)} mcm, y = -3.5 mm)")
            plt.savefig(f"./test_result/forces_crop_{int(rads[i]*1e6)}.png")
            plt.show()
            flag = False

        Fx_slice = Fx[15, :]
        Fy_slice = Fy[15, :]
        F_slice = np.sqrt(Fx_slice**2 + Fy_slice**2)
        plt.plot(x_clice, Fy_slice, ls= linestyles[i], c = colors[0], ms = 8, label = fr"$F_y$ (a = {int(rads[i]*1e6)} mcm)")
        plt.plot(x_clice, Fx_slice, ls= linestyles[i], c = colors[1], ms = 8, label = fr"$F_x$ (a = {int(rads[i]*1e6)} mcm)")
        plt.plot(x_clice, F_slice, ls= linestyles[i], c = colors[2], ms = 8, label = fr"$F$ (a = {int(rads[i]*1e6)} mcm)")
        print(f"Done: a = {int(rads[i]*1e6)} mcm")

    plt.xlabel("x, mm")
    plt.ylabel("F, N")
    plt.legend()
    plt.title("Forces in x-y direction (y = -3.5 mm)")
    plt.savefig(f"./test_result/forces_plot_comp.png")
    plt.show()    

def calculate_gorkov(a, field_z, Kx, Ky, params):
    rho = params.rho_medium
    rho_sp = params.rho_sph
    c_t = params.c_t
    c_l = params.c_l
    c = params.c_medium
    k = params.k_medium
    w = params.omega

    #Eq. (67)
    f1 = 1 - ((rho * c**2) / (rho_sp * c_l**2)) / (1 - (4 * c_t**2) / (3 * c_l**2))
    #Eq. (68)
    f2 = 2 * (rho_sp  - rho ) / (2 * rho_sp + rho)

    p_mean_sq = 0.5 * np.abs(field_z)**2

    S = np.fft.fft2(field_z)
    mask =  Kx**2 + Ky**2 <= k**2
    Kz = np.sqrt(np.where(mask, k**2 - (Kx**2 + Ky**2 ), 0))

    #Eq. (5)
    vx = np.fft.ifft2(Kx * S) / (w * rho)
    vy = np.fft.ifft2(Ky * S) / (w * rho)
    vz = np.fft.ifft2(Kz * S) / (w * rho)
    v_mean_sq = 0.5 * (np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2)

    #Eq. (79)
    U_gorkov = 2 * np.pi * a**3 * (f1 * p_mean_sq / (3 * rho * c**2) - f2 * rho * v_mean_sq / 2)
    U_spec = np.fft.fft2(U_gorkov)

    #Eq. (78)
    Fx = - np.real(np.fft.ifft2(1j * Kx * U_spec))
    Fy = - np.real(np.fft.ifft2(1j * Ky * U_spec))
    Fz = - np.real(np.fft.ifft2(1j * Kz * U_spec))

    return Fx, Fy, Fz, U_gorkov

def sapozh_gorkov_comparison(rad, X, Y, field_z, Kx, Ky, params):
    Nx = params.Nx
    Ny = params.Ny
    nx = params.nx
    ny = params.ny
    k = params.k_medium
    start_x = (Nx - nx) // 2
    start_y = (Ny - ny) // 2
    sl1 = (slice(start_y, start_y + ny), slice(start_x, start_x + nx))

    h1, h2 = 260, 330
    v1, v2 = 120, 190
    sl2 = (slice(h1, h2), slice(v1, v2))

    Y *= -1
    X_crop = X[sl1][sl2] * 1e3
    Y_crop = Y[sl1][sl2] * 1e3
    x_slice = X_crop[0, :]

    Fx, Fy, Fz, U = calculate_gorkov(rad, field_z, Kx, Ky, params)
    Fx, Fy, Fz, U = Fx[sl1][sl2], Fy[sl1][sl2], Fz[sl1][sl2], U[sl1][sl2]
    Fy *= -1

    Fx_slice = Fx[15, :]
    Fy_slice = Fy[15, :]
    F_slice = np.sqrt(Fx_slice**2 + Fy_slice**2)
    plt.plot(x_slice, Fx_slice, ls= '-', c = "#243bec", ms = 8, label = r"$F_x^{(g)}$")
    plt.plot(x_slice, F_slice, ls= '-', c = "#ec2424", ms = 8, label = r"$F^{(g)}$")
    
     
    Fx, Fy, Fz = calculate_forces(rad, field_z, Kx, Ky, params, n_max=get_nmax(k, rad), cropped=True, sl1 = sl1, sl2 = sl2)
    Fy *= -1

    Fx_slice = Fx[15, :]
    Fy_slice = Fy[15, :]
    F_slice = np.sqrt(Fx_slice**2 + Fy_slice**2)
    plt.plot(x_slice, Fx_slice, ls= '--', c = "#243bec", ms = 8, label = r"$F_x^{(s)}$")
    plt.plot(x_slice, F_slice, ls= '--', c = "#ec2424", ms = 8, label = r"$F^{(s)}$")

    plt.xlabel("x, mm")
    plt.ylabel("F, N")
    plt.legend()
    plt.title(f"Forces in x-y direction (y = -3.5 mm, ka = {round(k * rad, 3)})")
    plt.savefig(f"./test_result/forces_plot_gork_comp_{int(rad*1e6)}.png")
    plt.show()

def animate_particles(a, X, Y, field_z, Kx, Ky, params):
    Nx = params.Nx
    Ny = params.Ny
    dx = params.dx
    k = params.k_medium
    start_x = (Nx - params.nx) // 2
    start_y = (Ny - params.ny) // 2
    sl1 = (slice(start_y, start_y + params.ny), slice(start_x, start_x + params.nx))

    h1, h2 = 260, 330
    v1, v2 = 120, 190
    sl2 = (slice(h1, h2), slice(v1, v2))

    X_crop = X[sl1][sl2]
    Y_crop = - Y[sl1][sl2]
    Amp_crop = np.abs(field_z)[sl1][sl2]
    x_slice = X_crop[0, :]
    y_slice = Y_crop[:, 0][::-1]

    Fx, Fy, Fz = calculate_forces(a, field_z, Kx, Ky, params, n_max = get_nmax(k, a), cropped=True, sl1 = sl1, sl2 =sl2)
    Fy *= -1
    Fx = Fx[::-1, :]
    Fy = Fy[::-1, :]

    Fx_interpolator = RegularGridInterpolator((y_slice, x_slice), Fx, bounds_error=False, fill_value=0)
    Fy_interpolator = RegularGridInterpolator((y_slice, x_slice), Fy, bounds_error=False, fill_value=0)

    N_particles = 500
    pos = np.zeros((N_particles, 2))
    pos[:, 0] = np.random.uniform(np.min(x_slice), np.max(x_slice), N_particles)
    pos[:, 1] = np.random.uniform(np.min(y_slice), np.max(y_slice), N_particles)

    fig, ax = plt.subplots(figsize=(8, 8))

    extent = [np.min(x_slice) * 1e3, np.max(x_slice) * 1e3, np.min(y_slice) * 1e3, np.max(y_slice) * 1e3]
    im = ax.imshow(Amp_crop, cmap = 'jet', extent=extent, origin = 'upper', alpha=0.8)

    scat = plt.scatter(pos[:, 0] * 1e3, pos[:, 1] * 1e3, c = 'white', edgecolors='black', s = 25, zorder = 5)

    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")

    dt = 0.01 * (np.max(x_slice) - np.min(x_slice)) / np.max([np.max(np.abs(Fx)), np.max(np.abs(Fy))])

    def update(frame):

        points = np.column_stack((pos[:, 1], pos[:, 0]))

        Fx_int = Fx_interpolator(points)
        Fy_int = Fy_interpolator(points)

        pos[:, 0] += Fx_int * dt
        pos[:, 1] += Fy_int * dt

        scat.set_offsets(pos * 1e3)
        return scat, 
    
    ani = FuncAnimation(fig, update, frames = 100, interval = 10, blit = True)
    ani.save('./test_result/trap_animation.gif', writer='pillow', fps=30)

def main():
    params = Acousticparameters()
    dx = params.dx
    Nx = params.Nx
    Ny = params.Ny
    size_x = params.size_x
    size_y = params.size_y
    k = params.k_medium
    a = 50e-6

    X, Y, Kx, Ky = generate_grids(params)
    propagator = ASMPropagator(params, Kx, Ky, params.z_target)
    retrieved_phase = np.load('./test_result/retrieved_phase_dove.npy')

    p0_amplitude = create_source_amplitude(X, Y, radius = size_x/2)
    holo_field = p0_amplitude * np.exp(1j * retrieved_phase)
    field_z = propagator.forward(holo_field)
    field_z_amp = np.abs(field_z)

    Fx, Fy, Fz = calculate_forces(a, field_z, Kx, Ky, params, n_max=get_nmax(k, a))
    Fx_norm = Fx / np.max([np.max(np.abs(Fx)), np.max(np.abs(Fy))])
    Fy_norm = Fy / np.max([np.max(np.abs(Fx)), np.max(np.abs(Fy))])

    fig, ax = plt.subplots(figsize=(8,8))
    extent = [-Nx*dx/2 * 1e3, Nx*dx/2 * 1e3, -Ny*dx/2 * 1e3, Ny*dx/2 * 1e3]
    limit_x = params.size_x / 2 * 1e3
    limit_y = params.size_y / 2 * 1e3
    im = ax.imshow(field_z_amp, cmap = 'jet', extent=extent, origin='upper')
    fig.colorbar(im, ax=ax, shrink = 0.7)

    Y *= -1
    Fy_norm *= -1

    step = 4
    ax.quiver(X[::step, ::step] * 1e3, Y[::step, ::step] * 1e3, Fx_norm[::step, ::step], Fy_norm[::step, ::step], color = 'white', width = 0.003, scale=10)
    ax.set_xlim(-limit_x, limit_x)
    ax.set_ylim(-limit_y, limit_y)
    plt.xlabel("x, mm")
    plt.ylabel("y, mm")

    plt.savefig("./test_result/amp_dove.png")
    plt.show()
    
    force_vs_radius([50e-6, 75e-6, 100e-6], X, Y, field_z, Kx, Ky, params)
    sapozh_gorkov_comparison(1e-6, X, Y, field_z, Kx, Ky, params)
    animate_particles(50e-6, X, Y, field_z, Kx, Ky, params)
    
if __name__ == "__main__":
    main()

