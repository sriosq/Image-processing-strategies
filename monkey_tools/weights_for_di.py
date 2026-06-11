"""
Weights for DI .py
Implements the SEPIA v1 weighting pipeline for dipole field inversion.
Reference: https://sepia-documentation.readthedocs.io/en/latest/method/weightings.html

INPUT : noisesd.nii.gz  (fieldmap standard deviation map, 3D, calculated from n_std_for_bgfr.py in the same folder
        mask.nii.gz     (brain mask, 3D)
OUTPUT: weights.nii.gz
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import uniform_filter

# ---- 1. DEFINE INPUTS -------------------------------------------------------
noisesd_path = r"E:\msc_data\sc_qsm\neuropoly_data\chi_008\qsm_processing\n_std_output\n_std_T0000.nii.gz"
mask_path    = r"E:\msc_data\sc_qsm\neuropoly_data\chi_008\qsm_processing\chi_008_sc_msk.nii.gz"
out_path     = r"E:\msc_data\sc_qsm\neuropoly_data\chi_008\qsm_processing\n_std_output\weights.nii.gz"

# ---- 2. LOAD ----------------------------------------------------------------
sd_nii  = nib.load(noisesd_path)
noisesd = sd_nii.get_fdata().astype(np.float64)
print(noisesd.shape)

mask_nii = nib.load(mask_path)
mask     = mask_nii.get_fdata().astype(np.float64)
print(mask.shape)
# ---- 3. Invert fieldmap SD (eq. on Step 1) ----------------------------------
# weights = 1 / fieldmapSD
weights = np.zeros_like(noisesd)
nonzero = noisesd != 0
weights[nonzero] = 1.0 / noisesd[nonzero] 
weights[np.isnan(weights)] = 0
weights[np.isinf(weights)] = 0

# ---- 4. Normalise by median + 3*IQR in mask (eq. on Step 2) ----------------
in_mask      = weights[mask > 0]
median_w     = np.median(in_mask)
q75, q25     = np.percentile(in_mask, [75, 25])
iqr_w        = q75 - q25
norm_factor  = median_w + 3 * iqr_w
weights      = weights / norm_factor

# ---- 5. Re-centre median to 1 (eq. on Step 3) -------------------------------
# weights = weights - median(weights(mask)) + 1
in_mask2    = weights[mask > 0]
median_w2   = np.median(in_mask2)
weights     = weights - median_w2 + 1.0

# ---- 6. STEP 4: Clip outliers ---------------
# Original SEPIA replaces outliers with box-filtered values (3x3x3 kernel).
# Skipped here: 0.44x0.44x5mm voxels make that kernel do ~6x more smoothing
# in-plane than through-plane, and the cord is too thin for spatial blending
# near vessels to be safe. Hard clipping achieves the same outlier suppression
# without spatial assumptions
in_mask3     = weights[mask > 0]
median_w3    = np.median(in_mask3)
q75_3, q25_3 = np.percentile(in_mask3, [75, 25])
threshold    = median_w3 + 3 * (q75_3 - q25_3)

n_outliers = np.sum((weights > threshold) & (mask > 0))
print(f"\nOutlier voxels clipped: {n_outliers} / {int(np.sum(mask > 0))} in-mask voxels ({100*n_outliers/np.sum(mask>0):.2f}%)")

# Check this print, 

weights = np.clip(weights, 0, threshold)

# Zero out the provided mask
weights[mask == 0] = 0



# ---- 7. INSPECT -------------------------------------------------------------
in_mask_final = weights[mask > 0]
print("Weights stats (in-mask):")
print(f"  min    : {in_mask_final.min():.6f}")
print(f"  max    : {in_mask_final.max():.6f}")
print(f"  median : {np.median(in_mask_final):.6f}  (should be close to 1)")
print(f"  mean   : {in_mask_final.mean():.6f}")
print(f"  std    : {in_mask_final.std():.6f}")


# ---- 8. SAVE ----------------------------------------------------------------
out_nii = nib.Nifti1Image(weights, affine=sd_nii.affine, header=sd_nii.header)
out_nii.header.set_data_dtype(np.float64)
nib.save(out_nii, out_path)
print(f"\nSaved to: {out_path}")