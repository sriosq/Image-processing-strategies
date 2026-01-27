import numpy as np
# This is a python translation from the Julia code provided in:
# https://github.com/jisilva8/abdominal_qsm_phantom

import numpy as np

def dipole_kernel(shape, voxel_size, mode="continuous", b0_dir=(0, 0, 1)):
    """
    Dipole kernel for QSM (single orientation).
    
    Parameters
    ----------
    shape : tuple of int
        Image dimensions (Nx, Ny, Nz)
    voxel_size : tuple of float
        Voxel size in mm (dx, dy, dz)
    mode : str
        'continuous', 'discrete', 'green', 'integrated_green'
    b0_dir : tuple
        Main field direction (default z)
        
    Returns
    -------
    kernel : ndarray
        Dipole kernel in k-space (ifftshifted)
    """

    Nx, Ny, Nz = shape
    dx, dy, dz = voxel_size
    bx, by, bz = b0_dir

    # ---------- k-space coordinates ----------
    kx = np.fft.fftfreq(Nx, d=dx)
    ky = np.fft.fftfreq(Ny, d=dy)
    kz = np.fft.fftfreq(Nz, d=dz)

    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")

    k2 = kx**2 + ky**2 + kz**2
    k_dot_b0 = bx*kx + by*ky + bz*kz

    eps = np.finfo(float).eps

    # ---------- CONTINUOUS ----------
    if mode == "continuous":
        kernel = 1/3 - (k_dot_b0**2) / (k2 + eps)

    # ---------- DISCRETE (Milovic) ----------
    elif mode == "discrete":
        FOV = np.array(shape) * np.array(voxel_size)

        kx_d = 2*np.pi * kx * FOV[0]
        ky_d = 2*np.pi * ky * FOV[1]
        kz_d = 2*np.pi * kz * FOV[2]

        denom = (
            -3
            + np.cos(kx_d)
            + np.cos(ky_d)
            + np.cos(kz_d)
        )

        denom[denom == 0] = eps
        kernel = 1/3 - (-1 + np.cos(kz_d)) / denom

    # ---------- GREEN FUNCTION ----------
    elif mode == "green":
        r2 = kx**2 + ky**2 + kz**2
        kernel = (3*kz**2 - r2) / (4*np.pi * (r2**(5/2) + eps))
        kernel[r2 == 0] = 0.0

        kernel = np.real(np.fft.fftn(np.fft.ifftshift(kernel)))

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---------- DC handling ----------
    kernel[0, 0, 0] = 0.0

    return np.fft.ifftshift(kernel)
