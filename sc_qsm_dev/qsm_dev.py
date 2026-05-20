import numpy as np
import matplotlib.pyplot as plt


def lbv2d_python(total_field, mask, max_iter=10000, tol=1e-8, omega=1.7):
    """
    Simple 2D LBV-like solver.

    Solves Laplace(background) = 0 inside the mask,
    using total_field values as boundary conditions.
    Then local_field = total_field - background.
    """

    total_field = np.asarray(total_field, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    background = total_field.copy()

    nx, ny = total_field.shape

    for it in range(max_iter):
        max_change = 0.0

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):

                if not mask[i, j]:
                    continue

                # Do not update mask boundary voxels
                if (
                    not mask[i-1, j] or not mask[i+1, j] or
                    not mask[i, j-1] or not mask[i, j+1]
                ):
                    continue

                old = background[i, j]

                # 2D Laplace update
                new = 0.25 * (
                    background[i-1, j] +
                    background[i+1, j] +
                    background[i, j-1] +
                    background[i, j+1]
                )

                # SOR update
                background[i, j] = (1 - omega) * old + omega * new

                max_change = max(max_change, abs(background[i, j] - old))

        if max_change < tol:
            print(f"Converged after {it} iterations")
            break

    local_field = np.zeros_like(total_field)
    local_field[mask] = total_field[mask] - background[mask]

    return local_field, background

