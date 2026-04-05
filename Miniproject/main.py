import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from stl import mesh
from scipy.ndimage import zoom
from scipy.interpolate import make_interp_spline

from domain import generate_grids, create_source_amplitude
from parameters import Acousticparameters
from propagator import ASMPropagator
from iasa_solver import run_iasa, calculate_transmission, calculate_thickness

def export_stl(X, Y, phase, params, filename = "hologram.stl"):
    
    Nx = params.Nx
    Ny = params.Ny
    nx = params.nx + 10
    ny = params.ny + 10
    size_x = params.size_x
    start_x = (Nx - nx)//2
    start_y = (Ny - ny)//2

    radius = size_x / 2 * 1e3

    new_phase = phase[start_y : start_y + ny, start_x : start_x + nx]

    Tmm = calculate_thickness(new_phase, params) * 1e3
    dxmm = params.dx * 1e3
    
    x = (np.arange(nx) - nx/2) * dxmm
    y = (np.arange(ny) - ny/2) * dxmm
    X, Y = np.meshgrid(x, y)

    def get_point(j, i):
        x = X[j, i]
        y = Y[j, i]

        dist = np.sqrt(x**2 + y**2)

        if dist <= radius:
            p = [x, y, Tmm[j, i]]
        else:
            new_x = x * radius / dist
            new_y = y * radius / dist
            p = [new_x, new_y, params.T0 * 1e3]
        return p

    vectors=[]
    for i in range(nx - 1):
        for j in range(ny - 1):

            p1 = get_point(j, i)
            p2 = get_point(j, i+1)
            p3 = get_point(j+1, i)
            p4 = get_point(j+1, i+1)

            vectors.append(np.array([p1, p2, p3]))
            vectors.append(np.array([p2, p4, p3]))

    mesh_holo = mesh.Mesh(np.zeros(len(vectors), dtype=mesh.Mesh.dtype))
    for ind in range(len(vectors)):
        mesh_holo.vectors[ind] = np.array(vectors[ind])
    mesh_holo.save(filename)

def load_target_from_png(filepath, params, invert = False):
    Nx = params.Nx
    Ny = params.Ny
    nx = params.nx
    ny = params.ny

    img = Image.open(filepath).convert("L")
    img = img.resize((params.nx, params.ny), Image.Resampling.BILINEAR)

    target_pict = np.array(img, dtype = float)
    target_pict /= np.max(target_pict)
    target_pict = np.where(target_pict > 0.5, 1.0, 0.0)

    target_padded = np.zeros((Nx, Ny), dtype = float)
    start_x = (Nx - nx)//2
    start_y = (Ny - ny)//2
    target_padded[start_y : start_y + ny, start_x : start_x + nx] = target_pict

    if invert:
        target_padded = np.where(target_padded == 1.0, 0.0, 1.0)

    return target_padded

def z_scan(retrieved_phase, Kx, Ky, p0_amplitude, params, z_max = 50e-3, return_gif = False, num_frames = 50, filename = "./test_result/result.gif"):
    zs = np.linspace(0, z_max, num_frames)
    amplitudes = np.zeros((num_frames, params.Ny, params.Nx))

    alpha_t = calculate_transmission(retrieved_phase, params)
    holo_field = p0_amplitude * np.sqrt(alpha_t) * np.exp(1j * retrieved_phase)


    for i, z in enumerate(zs):
        propagator = ASMPropagator(params, Kx, Ky, z)

        field_z = propagator.forward(holo_field)
        amplitudes[i, :, :] = np.abs(field_z)

    if return_gif:
        fig, ax = plt.subplots(figsize=(7,6))
        extent = [-params.Nx*params.dx/2 * 1e3, params.Nx*params.dx/2 * 1e3, -params.Ny*params.dx/2 * 1e3, params.Ny*params.dx/2 * 1e3]
        limit_x = params.size_x / 2 * 1e3
        limit_y = params.size_y / 2 * 1e3

        im = ax.imshow(amplitudes[0], cmap = 'jet', extent = extent, vmin = 0, vmax = np.max(amplitudes))

        ax.set_xlabel("x, mm")
        ax.set_ylabel("y, mm")
        ax.set_xlim(-limit_x, limit_x)
        ax.set_ylim(-limit_y, limit_y)
        fig.colorbar(im, ax = ax, label = 'Pressure amplitude')

        def update(frame_index):
            im.set_array(amplitudes[frame_index])
            ax.set_title(f"Pressure amplitude (z = {int(zs[frame_index] * 1e3)} mm)")
            return [im]
        
        ani = animation.FuncAnimation(fig, update, frames = num_frames, blit = True)
        ani.save(filename, writer = "pillow", fps = 10)
        plt.close(fig)

    return amplitudes, zs

def plot_slice(amplitudes, zs , params, filename = "./test_result/scan_results.png"):
    pressure_slice = amplitudes[::-1, :, params.Nx // 2]
    plt.figure(figsize=(8, 8))

    extent = [- params.size_y / 2 * 1e3, params.size_y / 2 * 1e3, zs[0] * 1e3, zs[-1] * 1e3]
    plt.imshow(pressure_slice, cmap='jet', extent=extent)

    plt.colorbar(label = "Pressure amplitude")    
    plt.xlabel("y, mm")
    plt.ylabel("z, mm")
    plt.title("Pressure amplitude slice at x = 0 mm")

    plt.savefig(filename)
    plt.show()

def plot_phase_profile(phase, params, filename = "./test_result/phase_profile.png"):
    phase_slice = phase[ (params.Nx - params.nx) // 2: (params.Nx + params.nx) // 2, params.Nx // 2]
    x = np.linspace(0, params.size_x, params.nx)  * 1e3 - params.size_x * 1e3 / 2
    T = calculate_thickness(phase_slice, params) * 1e3
    np.savetxt('./test_result/phase_profile.csv', np.column_stack((x, T)), delimiter=',', fmt='%.6f')

    plt.plot(x, phase_slice)
    plt.xlabel("x, mm")
    plt.ylabel("phase, rad")
    plt.savefig(filename)
    plt.show()

def main():
    params = Acousticparameters()
    dx = params.dx
    Nx = params.Nx
    Ny = params.Ny
    size_x = params.size_x
    size_y = params.size_y


    X, Y, Kx, Ky = generate_grids(params)
    p0_amplitude = create_source_amplitude(X, Y, radius = size_x/2)

    image_path = "./test_data/dove.png"
    target_amplitude = load_target_from_png(image_path, params, invert = False)

    propagator = ASMPropagator(params, Kx, Ky, params.z_target)

    retrieved_phase = run_iasa(target_amplitude, p0_amplitude, propagator, params, iterations=40)
    alpha_t = calculate_transmission(retrieved_phase, params)

    holo_field = p0_amplitude * np.sqrt(alpha_t) * np.exp(1j * retrieved_phase)
    result_target_field = propagator.forward(holo_field)
    result_target_amp = np.abs(result_target_field)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    extent = [-Nx*dx/2 * 1e3, Nx*dx/2 * 1e3, -Ny*dx/2 * 1e3, Ny*dx/2 * 1e3]
    limit_x = params.size_x / 2 * 1e3
    limit_y = params.size_y / 2 * 1e3

    ax = axes[0]
    im = ax.imshow(target_amplitude, cmap='gray', extent=extent)
    ax.set_title("Target amplitude")
    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")
    ax.set_xlim(-limit_x, limit_x)
    ax.set_ylim(-limit_y, limit_y)
    fig.colorbar(im, ax=ax, shrink=0.7)

    ax = axes[1]
    phase_to_plot = np.where(p0_amplitude > 0, retrieved_phase, np.nan)
    im = ax.imshow(phase_to_plot, cmap='twilight', extent=extent)
    ax.set_title("Phase distribution")
    ax.set_xlabel("x, mm")
    ax.set_xlim(-limit_x, limit_x)
    ax.set_ylim(-limit_y, limit_y)
    fig.colorbar(im, ax=ax, shrink=0.7)


    ax = axes[2]
    im = ax.imshow(result_target_amp, cmap='jet', extent=extent)
    ax.set_title(f"Simulated amplitude (z = {int(params.z_target * 1e3)} mm)")
    ax.set_xlabel("x, mm")
    ax.set_xlim(-limit_x, limit_x)
    ax.set_ylim(-limit_y, limit_y)
    fig.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.savefig("./test_result/result_rings.png")
    plt.show()

    plot_phase_profile(retrieved_phase, params,filename="./test_result/phase_profile_rings.png")
    np.save('./test_result/retrieved_phase_dove.npy', retrieved_phase)

    export_stl(X, Y, retrieved_phase, params, "./test_result/hologram_.stl")

    amplitudes, zs = z_scan(retrieved_phase, Kx, Ky, p0_amplitude, params, filename = "./test_result/rings_scan.gif", return_gif=True)
    plot_slice(amplitudes, zs, params)

if __name__ == "__main__":
    main()




