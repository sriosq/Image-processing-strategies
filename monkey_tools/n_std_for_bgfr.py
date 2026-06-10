"""
Compute the Fieldmap SD 
Standalone script to compute the noise std map (N_std)
from magnitude images and echo times (TEs).
Based on Wrapper_TotalField_ROMEO.m from SEPIA toolbox
"""

import numpy as np
import nibabel as nib

# ---- 1. DEFINE INPUTS --------------------------------------------------------
mag_path  = r"E:\msc_data\sc_qsm\neuropoly_data\chi_014\chi_014\niftis\3D_5meGRE\chi_014_3D_5meGRE_mag.nii.gz"
mask_path = r"E:\msc_data\sc_qsm\neuropoly_data\chi_014\chi_014\niftis\3D_5meGRE\chi_014_3D_5meGRE_sc_msk.nii.gz"
out_path  = r"E:\msc_data\sc_qsm\neuropoly_data\chi_014\chi_014\qsm_processing\n_std_output\n_std.nii.gz"

TEs_vec = np.array([0.00693, 0.01185, 0.01685, 0.02185, 0.02685])  # seconds

# ---- 2. LOAD -----------------------------------------------------------------
mag_nii  = nib.load(mag_path)
mag      = mag_nii.get_fdata().astype(np.float64)   # [x, y, z, nEchoes]

mask_nii = nib.load(mask_path)
mask     = mask_nii.get_fdata().astype(np.float64)  # [x, y, z]

assert mag.shape[3] == len(TEs_vec), (
    f"Echo count mismatch: mag has {mag.shape[3]} echoes but {len(TEs_vec)} TEs given."
)

# ---- 3. RESHAPE TEs to broadcast across echo dimension ----------------------
# MATLAB: reshape(TEs_vec, 1,1,1,nEchoes)
TEs = TEs_vec[np.newaxis, np.newaxis, np.newaxis, :]  # [1, 1, 1, nEchoes]

# ---- 4. COMPUTE N_std -------------------------------------------------------
# MATLAB: sqrt(sum(mag .* mag .* (TEs .* TEs), 4))
N_std = np.sqrt(np.sum(mag**2 * TEs**2, axis=3))      # [x, y, z]

# ---- 5. INVERT --------------------------------------------------------------
N_std = 1.0 / N_std

# ---- 6. NORMALISE by norm of non-outlier in-mask voxels ---------------------
# MATLAB rmoutliers uses median ± 3*scaled MAD by default
in_mask = N_std[mask > 0]
median  = np.median(in_mask)
mad     = np.median(np.abs(in_mask - median))
threshold = 3 * 1.4826 * mad                          # scaled MAD, matches MATLAB default
cleaned = in_mask[np.abs(in_mask - median) <= threshold]
norm_factor = np.linalg.norm(cleaned)
N_std = N_std / norm_factor

# ---- 7. CLEAN UP NaN and Inf ------------------------------------------------
N_std = np.nan_to_num(N_std, nan=0.0, posinf=0.0, neginf=0.0)

# ---- 8. INSPECT -------------------------------------------------------------
in_mask_final = N_std[mask > 0]
print("N_std stats (in-mask):")
print(f"  min  : {in_mask_final.min():.6f}")
print(f"  max  : {in_mask_final.max():.6f}")
print(f"  mean : {in_mask_final.mean():.6f}")
print(f"  std  : {in_mask_final.std():.6f}")

# ---- 9. SAVE ----------------------------------------------------------------
out_nii = nib.Nifti1Image(N_std, affine=mag_nii.affine, header=mag_nii.header)
out_nii.header.set_data_dtype(np.float64)
nib.save(out_nii, out_path)
print(f"\nSaved to: {out_path}")