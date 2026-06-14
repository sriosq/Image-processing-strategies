"""
primordial_sepia_weights.py
Implements the pre-v1 SEPIA weighting pipeline for dipole field inversion.
Reference: https://sepia-documentation.readthedocs.io/en/latest/method/weightings.html
 
Two steps only:
  1. Invert N_std
  2. Normalise by max in-mask value (range 0-1)
 
Simpler than the SEPIA v1 histogram assumptions (brain-shaped, bimodal GM/WM) don't hold.
 
NOTE: All downstream algorithms (iLSQR, FANSI, MEDI) re-normalise internally
so the absolute scale here does not affect results — only relative voxel
values within the mask matter.
 
INPUT : n_std.nii.gz  (output of compute_n_std.py)
        mask.nii.gz   (spinal cord mask)
OUTPUT: weights_primordial.nii.gz
"""
 
import numpy as np
import nibabel as nib
 
# ---- 1. DEFINE INPUTS -------------------------------------------------------
noisesd_path = r"E:\msc_data\sc_qsm\neuropoly_data\chi_014\chi_014\qsm_processing\n_std_output\n_std.nii.gz"
mask_path    = r"E:\msc_data\sc_qsm\neuropoly_data\chi_014\chi_014\niftis\3D_5meGRE\chi_014_3D_5meGRE_sc_msk.nii.gz"
out_path     = r"E:\msc_data\sc_qsm\neuropoly_data\chi_014\chi_014\qsm_processing\n_std_output\weights_primordial.nii.gz"
 
# ---- 2. LOAD ----------------------------------------------------------------
sd_nii  = nib.load(noisesd_path)
noisesd = sd_nii.get_fdata().astype(np.float64)
 
mask_nii = nib.load(mask_path)
mask     = mask_nii.get_fdata().astype(np.float64)
 
# ---- 3. STEP 1: Invert N_std (eq. 12) --------------------------------------
# weights = 1 / N_std
weights = np.zeros_like(noisesd)
nonzero = noisesd != 0
weights[nonzero] = 1.0 / noisesd[nonzero]
weights[np.isnan(weights)] = 0
weights[np.isinf(weights)] = 0
 
# ---- 4. STEP 2: Normalise by max in-mask (eq. 13) --------------------------
# range becomes 0-1 within mask
in_mask            = weights[mask > 0]
weights[mask > 0]  = in_mask / in_mask.max()
weights[mask == 0] = 0
 
# ---- 5. INSPECT -------------------------------------------------------------
in_mask_final = weights[mask > 0]
print("Weights stats (in-mask):")
print(f"  min    : {in_mask_final.min():.6f}")
print(f"  max    : {in_mask_final.max():.6f}  (should be 1.0)")
print(f"  median : {np.median(in_mask_final):.6f}")
print(f"  mean   : {in_mask_final.mean():.6f}")
print(f"  std    : {in_mask_final.std():.6f}")
 
# ---- 6. SAVE ----------------------------------------------------------------
out_nii = nib.Nifti1Image(weights, affine=sd_nii.affine, header=sd_nii.header)
out_nii.header.set_data_dtype(np.float64)
nib.save(out_nii, out_path)
print(f"\nSaved to: {out_path}")