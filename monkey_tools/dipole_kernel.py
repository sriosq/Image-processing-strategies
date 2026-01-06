import numpy as np
# This is a python translation from the Julia code provided in:
# https://github.com/jisilva8/abdominal_qsm_phantom

def dipole_kernel(matrix_size, voxel_size, kernel_model=0):
    """
    Dipole kernel in k-space.

    Parameters
    ----------
    matrix_size : tuple of int
        (Nx, Ny, Nz)
    voxel_size : tuple of float
        (vx, vy, vz) in mm
    kernel_model : int
        0 -> Continuous kernel (Salomir et al. 2003)
        1 -> Discrete kernel (Milovic et al. 2017)

    Returns
    -------
    kernel : ndarray
        Dipole kernel in k-space (ifftshifted)
    """

    Nx, Ny, Nz = matrix_size
    vx, vy, vz = voxel_size

    eps = np.finfo(np.float32).eps

    # ============================================================
    # Continuous kernel model (Salomir et al. 2003)
    # ============================================================
    if kernel_model == 0:

        ky_range = np.arange(-np.floor(Nx/2), np.ceil(Nx/2))
        kx_range = np.arange(-np.floor(Ny/2), np.ceil(Ny/2))
        kz_range = np.arange(-np.floor(Nz/2), np.ceil(Nz/2))

        ky, kx, kz = np.meshgrid(
            ky_range, kx_range, kz_range, indexing='ij'
        )

        # Normalize and scale k-space
        kx = kx.astype(np.float32)
        ky = ky.astype(np.float32)
        kz = kz.astype(np.float32)

        kx = (kx / np.max(np.abs(kx))) / vx
        ky = (ky / np.max(np.abs(ky))) / vy
        kz = (kz / np.max(np.abs(kz))) / vz

        # k^2 term
        k2 = kx**2 + ky**2 + kz**2
        k2[k2 == 0] = eps

        # Dipole kernel
        kernel = 1/3 - (kz**2) / k2

    # ============================================================
    # Discrete kernel model (Milovic et al. 2017)
    # ============================================================
    else:

        cx = 1 + Nx // 2
        cy = 1 + Ny // 2
        cz = 1 + Nz // 2

        kx = np.arange(1, Nx+1) - cx
        ky = np.arange(1, Ny+1) - cy
        kz = np.arange(1, Nz+1) - cz

        dx, dy, dz = 1 / (np.array(matrix_size) * np.array(voxel_size))

        kx = kx * dx
        ky = ky * dy
        kz = kz * dz

        kx = kx.reshape(-1, 1, 1)
        ky = ky.reshape(1, -1, 1)
        kz = kz.reshape(1, 1, -1)

        kx = np.repeat(kx, Ny, axis=1)
        kx = np.repeat(kx, Nz, axis=2)

        ky = np.repeat(ky, Nx, axis=0)
        ky = np.repeat(ky, Nz, axis=2)

        kz = np.repeat(kz, Nx, axis=0)
        kz = np.repeat(kz, Ny, axis=1)

        # Discrete Laplacian term
        k2 = -3 + np.cos(2*np.pi*kx) + np.cos(2*np.pi*ky) + np.cos(2*np.pi*kz)
        k2[k2 == 0] = eps

        # Dipole kernel
        kernel = 1/3 - (-1 + np.cos(2*np.pi*kz)) / k2

    # ============================================================
    # Shift kernel so DC is at [0,0,0]
    # ============================================================
    kernel = np.fft.ifftshift(kernel)
    kernel[0, 0, 0] = 0.0

    return kernel
