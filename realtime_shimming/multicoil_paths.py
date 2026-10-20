"""Shared native-space data paths for the eight-participant multicoil study.

Place this file beside the magnitude/fieldmap, BGFR, and DI notebooks.
Edit the explicit path variables below; missing or optional files use r"".
Keep filenames exactly as exported by the toolbox. All known paths were
copied from the original notebook and have not been checked on the R: drive.

Notebook usage (rerun after saving path edits):
    import importlib
    import multicoil_paths as paths
    importlib.reload(paths)
    participants = paths.participants

Measured conditions: sh (SH), sh_acdc (SH+ACDC).
Predicted condition: predicted_sh_acdc (optional).
Local-field and susceptibility analyses use NIfTIs and matching masks,
not precomputed CSVs. PAM50-space outputs should be kept separately from
these native-space paths when their registration output layout is settled.
"""

CONDITION_LABELS = {
    "sh": "SH",
    "sh_acdc": "SH+ACDC",
    "predicted_sh_acdc": "Predicted SH+ACDC",
}
MEASURED_CONDITIONS = ("sh", "sh_acdc")
PARTICIPANT_IDS = tuple(f"sub-{number:02d}" for number in range(1, 9))

# EDITABLE PATHS — fill missing entries below as processing finishes
# Multicoil acquisitions only. baseline = SH; shimmed = SH+ACDC.
# Existing paths were copied from the old notebook and are not existence-verified.
# Empty strings mark missing/unconfirmed paths. No Siemens-coil paths are included.
# Only total-fieldmap metrics use CSV paths. Local-field and susceptibility metrics
# are calculated from NIfTIs using the matching SC/GM/WM masks and vertebral labels.
# Masks/labels must share the analyzed image grid; OPT PDF may require a validity mask.
# The predicted field is optional and distinct from the measured SH+ACDC field.

# ==================== sub-01 // chhi_026 ====================
# SH shim
sub01_baseline_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\sc-qsm-output\chi_026_baseline\masking\chi_026_baseline_echo-avg_magnitude.nii.gz"
sub01_baseline_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\chi_026_3D_baselineShim_5megre_C3_3_sc_msk.nii.gz"  # Spinal-cord mask NIfTI
sub01_baseline_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\sc-qsm-output\chi_026_baseline\masking\chi_026_baseline_echo-avg_desc-GM_mask.nii.gz"
sub01_baseline_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\sc-qsm-output\chi_026_baseline\masking\chi_026_baseline_echo-avg_desc-WM_mask.nii.gz"
sub01_baseline_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\sc-qsm-output\chi_026_baseline\masking\register\chi_026_baseline_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub01_baseline_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\B0_baseline_masked.nii.gz"  # Measured total fieldmap NIfTI, Hz
sub01_baseline_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\sct_metrics\chi_026_baseline_fieldmap_slicewise_metrics.csv"  # Total fieldmap slice-wise metrics CSV
sub01_baseline_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\sc-qsm-output\chi_026_baseline\bgfr\optimized\pdf\chi_026_baseline_desc-localfield.nii.gz" # Hz
sub01_baseline_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\sc-qsm-output\chi_026_baseline\dipole_inversion\optimized\closedForm\from_PDF_optimized\chi_026_baseline_desc-chimap.nii.gz" #ppm
sub01_baseline_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\sc-qsm-output\chi_026_baseline\dipole_inversion\optimized\medi\from_PDF_optimized\chi_026_baseline_desc-chimap.nii.gz"
sub01_baseline_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\sc-qsm-output\chi_026_baseline\dipole_inversion\optimized\fansi\from_PDF_optimized\chi_026_baseline_desc-chimap.nii.gz"
# SH+ACDC
sub01_shimmed_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\masking\chi_026_3D_MCShim_5megre_C3_mag_averaged.nii.gz"  # Echo-average magnitude NIfTI
sub01_shimmed_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\masking\chi_026_mc_no_reg_desc-SC_mask.nii.gz"  # Spinal-cord mask NIfTI
sub01_shimmed_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\masking\chi_026_mc_no_reg_desc-GM_mask.nii.gz"  # Gray-matter mask NIfTI
sub01_shimmed_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\masking\chi_026_mc_no_reg_desc-WM_mask.nii.gz"  # White-matter mask NIfTI
sub01_shimmed_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\masking\register\chi_026_mc_no_reg_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"  # Vertebral-level segmentation in the analysis image space
sub01_shimmed_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\fieldmap\masked_fieldmap\chi_026_mc_no_reg_desc-b0_fieldmap.nii.gz"  # Measured total fieldmap NIfTI, Hz
sub01_shimmed_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\sct_metrics\chi_026_mc_noreg_fieldmap_slicewise_metrics.csv"  # Total fieldmap slice-wise metrics CSV
sub01_shimmed_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\bgfr\comp_bgfr\opt_pdf\chi_026_mc_no_reg_desc-localfield.nii.gz" # Hz
sub01_shimmed_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\dipole_inversion\optimized\closedForm\from_PDF_optimized\chi_026_mc_no_reg_desc-chimap.nii.gz"
sub01_shimmed_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\dipole_inversion\optimized\medi\from_PDF_optimized\chi_026_mc_no_reg_desc-chimap.nii.gz"
sub01_shimmed_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\no_reg\pp_gui_dev\chi_026_mc_no_reg\dipole_inversion\optimized\fansi\from_PDF_optimized\chi_026_mc_no_reg_desc-chimap.nii.gz"
# Optional prediction of SH+ACDC from the SH acquisition
sub01_baseline_predict_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\raw_data\chi_026\chi_026\acdc_shimm_experiment_dcms\rt_shim_nifti_no_reg\shim_coeff\masked_fm_calculated_shim.nii.gz"
sub01_baseline_predict_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_26\qsm_processing\mc\baseline\chi_026_st_predict_fieldmap_slicewise_metrics.csv"

# ==================== sub-02 // chi_30 ====================
# SH shim
sub02_baseline_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\masking\sub-19_mc_r1_bl_echo-avg_magnitude.nii.gz"
sub02_baseline_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\masking\sub-19_mc_r1_bl_echo-avg_desc-SC_mask.nii.gz"  # Spinal-cord mask NIfTI
sub02_baseline_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\masking\sub-19_mc_r1_bl_echo-avg_desc-GM_mask.nii.gz"
sub02_baseline_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\masking\sub-19_mc_r1_bl_echo-avg_desc-WM_mask.nii.gz"
sub02_baseline_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\masking\register\sub-19_mc_r1_bl_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub02_baseline_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\fieldmap\masked_fieldmap\sub-19_mc_r1_bl_desc-b0_fieldmap.nii.gz"  # Measured total fieldmap NIfTI, Hz
sub02_baseline_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\sct_metrics\chi_030_baseline_fieldmap_slicewise_metrics.csv"  # Total fieldmap slice-wise metrics CSV
sub02_baseline_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\bgfr\comp_bgfr\opt_pdf\sub-19_mc_r1_bl_desc-localfield.nii.gz"
sub02_baseline_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\dipole_inversion\comp_di\opt_closedForm\sub-19_mc_r1_bl_desc-chimap.nii.gz"
sub02_baseline_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\dipole_inversion\comp_di\opt_medi\sub-19_mc_r1_bl_desc-chimap.nii.gz"
sub02_baseline_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\baseline\sc-qsm-output\sub-19_mc_r1_bl\dipole_inversion\comp_di\opt_fansi\sub-19_mc_r1_bl_desc-chimap.nii.gz"
# SH+ACDC
sub02_shimmed_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\masking\sub-19_mc_r1_shim_echo-avg_magnitude.nii.gz"  # Echo-average magnitude NIfTI
sub02_shimmed_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\masking\sub-19_mc_r1_shim_echo-avg_desc-SC_mask.nii.gz"  # Spinal-cord mask NIfTI
sub02_shimmed_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\masking\sub-19_mc_r1_shim_echo-avg_desc-GM_mask.nii.gz"  # Gray-matter mask NIfTI
sub02_shimmed_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\masking\sub-19_mc_r1_shim_echo-avg_desc-WM_mask.nii.gz"  # White-matter mask NIfTI
sub02_shimmed_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\masking\register\sub-19_mc_r1_shim_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"  # Vertebral-level segmentation in the analysis image space
sub02_shimmed_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\fieldmap\masked_fieldmap\sub-19_mc_r1_shim_desc-b0_fieldmap.nii.gz"  # Measured total fieldmap NIfTI, Hz
sub02_shimmed_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\sct_metrics\chi_030_test1_mc_shim_fieldmap_slicewise_metrics.csv"  # Total fieldmap slice-wise metrics CSV
sub02_shimmed_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\bgfr\comp_bgfr\opt_pdf\sub-19_mc_r1_shim_desc-localfield.nii.gz"
sub02_shimmed_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\dipole_inversion\comp_di\opt_closedForm\sub-19_mc_r1_shim_desc-chimap.nii.gz"
sub02_shimmed_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\dipole_inversion\comp_di\opt_medi\sub-19_mc_r1_shim_desc-chimap.nii.gz"
sub02_shimmed_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\qsm_processing\mc\test1\shimmed\sc-qsm-output\sub-19_mc_r1_shim\dipole_inversion\comp_di\opt_fansi\sub-19_mc_r1_shim_desc-chimap.nii.gz"
# Optional prediction of SH+ACDC from the SH acquisition
sub02_baseline_predict_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\st_predict_test1\fieldmap_calculated_shim.nii.gz"
sub02_baseline_predict_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_030\st_predict_test1\chi_030_test1_st_predict_fieldmap_slicewise_metrics.csv"

# ==================== sub-03 // chi_032 ====================
# SH shim
sub03_baseline_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\masking\chi_032_r1_bl_echo-avg_magnitude.nii.gz"
sub03_baseline_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\masking\chi_032_r1_bl_echo-avg_desc-SC_mask.nii.gz"  # Spinal-cord mask NIfTI
sub03_baseline_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\masking\chi_032_r1_bl_echo-avg_desc-GM_mask.nii.gz"
sub03_baseline_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\masking\chi_032_r1_bl_echo-avg_desc-WM_mask.nii.gz"
sub03_baseline_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\masking\register\chi_032_r1_bl_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub03_baseline_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\fieldmap\masked_fieldmap\chi_032_r1_bl_desc-b0_fieldmap.nii.gz"  # Measured total fieldmap NIfTI, Hz
sub03_baseline_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\sct_metrics\fieldmap_slicewise_metrics.csv"  # Total fieldmap slice-wise metrics CSV
sub03_baseline_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\bgfr\optimized\opt_pdf\chi_032_r1_bl_desc-localfield.nii.gz"
sub03_baseline_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\dipole_inversion\optimized\closedForm\from_PDF_optimized\chi_032_r1_bl_desc-chimap.nii.gz"
sub03_baseline_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\dipole_inversion\optimized\medi\from_PDF_optimized\chi_032_r1_bl_desc-chimap.nii.gz"
sub03_baseline_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\baseline1\sc-qsm-output\chi_032_r1_bl\dipole_inversion\optimized\fansi\from_PDF_optimized\chi_032_r1_bl_desc-chimap.nii.gz"
# SH+ACDC
sub03_shimmed_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\masking\chi_032_r1_shim_echo-avg_magnitude.nii.gz"  # Echo-average magnitude NIfTI
sub03_shimmed_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\masking\chi_032_r1_shim_echo-avg_desc-SC_mask.nii.gz"  # Spinal-cord mask NIfTI
sub03_shimmed_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\masking\chi_032_r1_shim_echo-avg_desc-GM_mask.nii.gz"  # Gray-matter mask NIfTI
sub03_shimmed_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\masking\chi_032_r1_shim_echo-avg_desc-WM_mask.nii.gz"  # White-matter mask NIfTI
sub03_shimmed_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\masking\register\chi_032_r1_shim_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub03_shimmed_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\fieldmap\masked_fieldmap\chi_032_r1_shim_desc-b0_fieldmap.nii.gz"  # Measured total fieldmap NIfTI, Hz
sub03_shimmed_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\sct_metrics\fieldmap_slicewise_metrics.csv"  # Total fieldmap slice-wise metrics CSV
sub03_shimmed_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\bgfr\optimized\opt_pdf\chi_032_r1_shim_desc-localfield.nii.gz"
sub03_shimmed_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\dipole_inversion\optimized\closedForm\from_PDF_optimized\chi_032_r1_shim_desc-chimap.nii.gz"
sub03_shimmed_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\dipole_inversion\optimized\medi\from_PDF_optimized\chi_032_r1_shim_desc-chimap.nii.gz"
sub03_shimmed_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\qsm_processing\test1\mcshim1\sc-qsm-output\chi_032_r1_shim\dipole_inversion\optimized\fansi\from_PDF_optimized\chi_032_r1_shim_desc-chimap.nii.gz"
# Optional prediction of SH+ACDC from the SH acquisition
sub03_baseline_predict_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\dcms_and_niftis\niftis\rt_shim_nifti_test1\shim_coeff\fieldmap_calculated_shim.nii.gz"
sub03_baseline_predict_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_032\dcms_and_niftis\niftis\rt_shim_nifti_test1\shim_coeff\test1_pred_fieldmap_slicewise_metrics.csv"

# ==================== sub-04 // chi_033 ====================
# SH shim
sub04_baseline_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\masking\chi_033_r1_bl_echo-avg_magnitude.nii.gz"
sub04_baseline_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\masking\chi_033_r1_bl_echo-avg_desc-SC_mask.nii.gz"  # Spinal-cord mask NIfTI
sub04_baseline_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\masking\chi_033_r1_bl_echo-avg_desc-GM_mask.nii.gz"
sub04_baseline_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\masking\chi_033_r1_bl_echo-avg_desc-WM_mask.nii.gz"
sub04_baseline_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\masking\register\chi_033_r1_bl_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub04_baseline_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\fieldmap\masked_fieldmap\chi_033_r1_bl_desc-b0_fieldmap.nii.gz"  # Measured total fieldmap NIfTI, Hz
sub04_baseline_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\sct_metrics\chi_033_test1_baseline_fieldmap_slicewise_metrics.csv"  # Total fieldmap slice-wise metrics CSV
sub04_baseline_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\bgfr\comp_bgfr\opt_pdf\chi_033_r1_bl_desc-localfield.nii.gz"
sub04_baseline_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\dipole_inversion\optimized\closedForm\from_PDF_optimized\chi_033_r1_bl_desc-chimap.nii.gz"
sub04_baseline_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\dipole_inversion\optimized\medi\from_PDF_optimized\chi_033_r1_bl_desc-chimap.nii.gz"
sub04_baseline_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\baseline1\sc-qsm-output\chi_033_r1_bl\dipole_inversion\optimized\fansi\from_PDF_optimized\chi_033_r1_bl_desc-chimap.nii.gz"
# SH+ACDC
sub04_shimmed_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\masking\chi_033_r1_shim_echo-avg_magnitude.nii.gz"  # Echo-average magnitude NIfTI
sub04_shimmed_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\masking\chi_033_r1_shim_echo-avg_desc-SC_mask.nii.gz"  # Spinal-cord mask NIfTI
sub04_shimmed_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\masking\chi_033_r1_shim_echo-avg_desc-GM_mask.nii.gz"  # Gray-matter mask NIfTI
sub04_shimmed_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\masking\chi_033_r1_shim_echo-avg_desc-WM_mask.nii.gz"  # White-matter mask NIfTI
sub04_shimmed_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\masking\register\chi_033_r1_shim_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub04_shimmed_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\fieldmap\masked_fieldmap\chi_033_r1_shim_desc-b0_fieldmap.nii.gz"  # Measured total fieldmap NIfTI, Hz
sub04_shimmed_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\sct_metrics\chi_033_test1_mc_shim_fieldmap_slicewise_metrics.csv"  # Total fieldmap slice-wise metrics CSV
sub04_shimmed_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\bgfr\comp_bgfr\opt_pdf\chi_033_r1_shim_desc-localfield.nii.gz"
sub04_shimmed_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\dipole_inversion\optimized\closedForm\from_PDF_optimized\chi_033_r1_shim_desc-chimap.nii.gz"
sub04_shimmed_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\dipole_inversion\optimized\medi\from_PDF_optimized\chi_033_r1_shim_desc-chimap.nii.gz"
sub04_shimmed_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\qsm_processing\mc_test1\shimmed1\sc-qsm-output\chi_033_r1_shim\dipole_inversion\optimized\fansi\from_PDF_optimized\chi_033_r1_shim_desc-chimap.nii.gz"
# Optional prediction of SH+ACDC from the SH acquisition
sub04_baseline_predict_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\dcms_and_niftis\chi_033\chi_033\rt_shim_nifti\shim_coeff\fieldmap_calculated_shim.nii.gz"
sub04_baseline_predict_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_033\dcms_and_niftis\chi_033\chi_033\rt_shim_nifti\shim_coeff\chi_033_test1_st_predict_fieldmap_slicewise_metrics.csv"

# ==================== sub-05 ====================
# SH shim
sub05_baseline_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\masking\chi-035_3D_bl_echo-avg_magnitude.nii.gz"
sub05_baseline_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\masking\chi-035_3D_bl_echo-avg_desc-SC_mask.nii.gz"
sub05_baseline_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\masking\chi-035_3D_bl_echo-avg_desc-GM_mask.nii.gz"
sub05_baseline_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\masking\chi-035_3D_bl_echo-avg_desc-WM_mask.nii.gz"
sub05_baseline_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\masking\register\chi-035_3D_bl_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub05_baseline_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\fieldmap\masked_fieldmap\chi-035_3D_bl_desc-b0_fieldmap.nii.gz"
sub05_baseline_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\sct_metrics\fieldmap_slicewise_metrics.csv"
sub05_baseline_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\bgfr\comp_bgfr\opt_pdf\chi-035_3D_bl_desc-localfield.nii.gz"
sub05_baseline_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\dipole_inversion\comp_di\opt_closedForm\chi-035_3D_bl_desc-chimap.nii.gz"
sub05_baseline_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\dipole_inversion\comp_di\opt_medi\chi-035_3D_bl_desc-chimap.nii.gz"
sub05_baseline_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\baseline\sc-qsm-output\chi-035_3D_bl\dipole_inversion\comp_di\opt_fansi\chi-035_3D_bl_desc-chimap.nii.gz"
# SH+ACDC
sub05_shimmed_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\masking\chi-035_3D_sh_echo-avg_magnitude.nii.gz"
sub05_shimmed_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\masking\chi-035_3D_sh_echo-avg_desc-SC_mask.nii.gz"
sub05_shimmed_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\masking\chi-035_3D_sh_echo-avg_desc-GM_mask.nii.gz"
sub05_shimmed_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\masking\chi-035_3D_sh_echo-avg_desc-WM_mask.nii.gz"
sub05_shimmed_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\masking\register\chi-035_3D_sh_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub05_shimmed_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\fieldmap\masked_fieldmap\chi-035_3D_sh_desc-b0_fieldmap.nii.gz"
sub05_shimmed_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\sct_metrics\fieldmap_slicewise_metrics.csv"
sub05_shimmed_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\bgfr\comp_bgfr\opt_pdf\chi-035_3D_sh_desc-localfield.nii.gz"
sub05_shimmed_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\dipole_inversion\comp_di\closedForm\optimized\from_PDF_optimized\chi-035_3D_sh_desc-chimap.nii.gz"
sub05_shimmed_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\dipole_inversion\comp_di\medi\optimized\from_PDF_optimized\chi-035_3D_sh_desc-chimap.nii.gz"
sub05_shimmed_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\qsm_processing\3D\shim\sc-qsm-output\chi-035_3D_sh\dipole_inversion\comp_di\fansi\optimized\from_PDF_optimized\chi-035_3D_sh_desc-chimap.nii.gz"
# Optional prediction of SH+ACDC from the SH acquisition
sub05_baseline_predict_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\dcms_and_niftis\raw_after_scan\rt_shim_nifti_3D_MC\shim_coeff\fieldmap_calculated_shim.nii.gz"
sub05_baseline_predict_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_035\dcms_and_niftis\raw_after_scan\rt_shim_nifti_3D_MC\shim_coeff\chi_035_st_predict_fieldmap_slicewise_metrics.csv"

# ==================== sub-06 ====================
# SH shim
sub06_baseline_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\masking\chi-036_3D_bl_echo-avg_magnitude.nii.gz"
sub06_baseline_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\masking\chi-036_3D_bl_echo-avg_desc-SC_mask.nii.gz"
sub06_baseline_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\masking\chi-036_3D_bl_echo-avg_desc-GM_mask.nii.gz"
sub06_baseline_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\masking\chi-036_3D_bl_echo-avg_desc-WM_mask.nii.gz"
sub06_baseline_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\masking\register\chi-036_3D_bl_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub06_baseline_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\fieldmap\masked_fieldmap\chi-036_3D_bl_desc-b0_fieldmap.nii.gz"
sub06_baseline_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\sct_metrics\fieldmap_slicewise_metrics.csv"
sub06_baseline_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\bgfr\comp_bgfr\opt_pdf\chi-036_3D_bl_desc-localfield.nii.gz"
sub06_baseline_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\dipole_inversion\optimized\closedForm\from_PDF_optimized\chi-036_3D_bl_desc-chimap.nii.gz"
sub06_baseline_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\dipole_inversion\comp_di\medi\optimized\from_PDF_optimized\chi-036_3D_bl_desc-chimap.nii.gz"
sub06_baseline_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\baseline\sc-qsm-output\chi-036_3D_bl\dipole_inversion\comp_di\fansi\optimized\from_PDF_optimized\chi-036_3D_bl_desc-chimap.nii.gz"
# SH+ACDC
sub06_shimmed_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\masking\chi-036_3D_sh_echo-avg_magnitude.nii.gz"
sub06_shimmed_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\masking\chi-036_3D_sh_echo-avg_desc-SC_mask.nii.gz"
sub06_shimmed_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\masking\chi-036_3D_sh_echo-avg_desc-GM_mask.nii.gz"
sub06_shimmed_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\masking\chi-036_3D_sh_echo-avg_desc-WM_mask.nii.gz"
sub06_shimmed_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\masking\register\chi-036_3D_sh_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub06_shimmed_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\fieldmap\masked_fieldmap\chi-036_3D_sh_desc-b0_fieldmap.nii.gz"
sub06_shimmed_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\sct_metrics\fieldmap_slicewise_metrics.csv"
sub06_shimmed_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\bgfr\comp_bgfr\opt_pdf\chi-036_3D_sh_desc-localfield.nii.gz"
sub06_shimmed_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\dipole_inversion\comp_di\closedForm\optimized\from_PDF_optimized\chi-036_3D_sh_desc-chimap.nii.gz"
sub06_shimmed_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\dipole_inversion\comp_di\medi\optimized\from_PDF_optimized\chi-036_3D_sh_desc-chimap.nii.gz"
sub06_shimmed_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\qsm_processing\3D\shim\sc-qsm-output\chi-036_3D_sh\dipole_inversion\comp_di\fansi\optimized\from_PDF_optimized\chi-036_3D_sh_desc-chimap.nii.gz"
# Optional prediction of SH+ACDC from the SH acquisition
sub06_baseline_predict_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\dcms_and_niftis\dcms_and_mac_processed\rt_shim_nifti_3D\shim_coeff\fieldmap_calculated_shim.nii.gz"
sub06_baseline_predict_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_036\dcms_and_niftis\dcms_and_mac_processed\rt_shim_nifti_3D\shim_coeff\chi_036_st_predict_fieldmap_slicewise_metrics.csv"

# ==================== sub-07 ====================
# SH shim
sub07_baseline_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\masking\chi-037_3D_bl_echo-avg_magnitude.nii.gz"
sub07_baseline_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\masking\chi-037_3D_bl_echo-avg_desc-SC_mask.nii.gz"
sub07_baseline_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\masking\chi-037_3D_bl_echo-avg_desc-GM_mask.nii.gz"
sub07_baseline_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\masking\chi-037_3D_bl_echo-avg_desc-WM_mask.nii.gz"
sub07_baseline_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\masking\register\chi-037_3D_bl_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub07_baseline_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\fieldmap\masked_fieldmap\chi-037_3D_bl_desc-b0_fieldmap.nii.gz"
sub07_baseline_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\sct_metrics\fieldmap_slicewise_metrics.csv"
sub07_baseline_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\bgfr\comp_bgfr\opt_pdf\chi-037_3D_bl_desc-localfield.nii.gz"
sub07_baseline_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\dipole_inversion\comp_di\closedForm\optimized\from_PDF_optimized\chi-037_3D_bl_desc-chimap.nii.gz"
sub07_baseline_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\dipole_inversion\comp_di\medi\optimized\from_PDF_optimized\chi-037_3D_bl_desc-chimap.nii.gz"
sub07_baseline_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\baseline\sc-qsm-output\chi-037_3D_bl\dipole_inversion\comp_di\fansi\optimized\from_PDF_optimized\chi-037_3D_bl_desc-chimap.nii.gz"
# SH+ACDC
sub07_shimmed_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\masking\chi-037_3D_sh_echo-avg_magnitude.nii.gz"
sub07_shimmed_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\masking\chi-037_3D_sh_echo-avg_desc-SC_mask.nii.gz"
sub07_shimmed_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\masking\chi-037_3D_sh_echo-avg_desc-GM_mask.nii.gz"
sub07_shimmed_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\masking\chi-037_3D_sh_echo-avg_desc-WM_mask.nii.gz"
sub07_shimmed_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\masking\register\chi-037_3D_sh_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub07_shimmed_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\fieldmap\masked_fieldmap\chi-037_3D_sh_desc-b0_fieldmap.nii.gz"
sub07_shimmed_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\sct_metrics\fieldmap_slicewise_metrics.csv"
sub07_shimmed_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\bgfr\comp_bgfr\opt_pdf\chi-037_3D_sh_desc-localfield.nii.gz"
sub07_shimmed_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\dipole_inversion\comp_di\closedForm\optimized\from_PDF_optimized\chi-037_3D_sh_desc-chimap.nii.gz"
sub07_shimmed_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\dipole_inversion\comp_di\medi\optimized\from_PDF_optimized\chi-037_3D_sh_desc-chimap.nii.gz"
sub07_shimmed_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\qsm_processing\3D\shim\sc-qsm-output\chi-037_3D_sh\dipole_inversion\comp_di\fansi\optimized\from_PDF_optimized\chi-037_3D_sh_desc-chimap.nii.gz"
# Optional prediction of SH+ACDC from the SH acquisition
sub07_baseline_predict_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\dcms_and_niftis\dcms_and_mac_processed\rt_shim_nifti_3D\shim_coeff\fieldmap_calculated_shim.nii.gz"
sub07_baseline_predict_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_037\dcms_and_niftis\dcms_and_mac_processed\rt_shim_nifti_3D\shim_coeff\chi_037_st_predict_fieldmap_slicewise_metrics.csv"

# ==================== sub-08 ====================
# SH shim
sub08_baseline_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\masking\chi-038_3D_bl_echo-avg_magnitude.nii.gz"
sub08_baseline_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\masking\chi-038_3D_bl_echo-avg_desc-SC_mask.nii.gz"
sub08_baseline_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\masking\chi-038_3D_bl_echo-avg_desc-GM_mask.nii.gz"
sub08_baseline_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\masking\chi-038_3D_bl_echo-avg_desc-WM_mask.nii.gz"
sub08_baseline_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\masking\register\chi-038_3D_bl_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub08_baseline_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\fieldmap\masked_fieldmap\chi-038_3D_bl_desc-b0_fieldmap.nii.gz"
sub08_baseline_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\sct_metrics\fieldmap_slicewise_metrics.csv"
sub08_baseline_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\bgfr\comp_bgfr\opt_pdf\chi-038_3D_bl_desc-localfield.nii.gz"
sub08_baseline_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\dipole_inversion\comp_di\closedForm\optimized\from_PDF_optimized\chi-038_3D_bl_desc-chimap.nii.gz"
sub08_baseline_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\dipole_inversion\comp_di\medi\optimized\from_PDF_optimized\chi-038_3D_bl_desc-chimap.nii.gz"
sub08_baseline_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\baseline\sc-qsm-output\chi-038_3D_bl\dipole_inversion\comp_di\fansi\optimized\from_PDF_optimized\chi-038_3D_bl_desc-chimap.nii.gz"
# SH+ACDC
sub08_shimmed_magnitude_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\masking\chi-038_3D_sh_echo-avg_magnitude.nii.gz"
sub08_shimmed_sc_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\masking\chi-038_3D_sh_echo-avg_desc-SC_mask.nii.gz"
sub08_shimmed_gm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\masking\chi-038_3D_sh_echo-avg_desc-GM_mask.nii.gz"
sub08_shimmed_wm_mask_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\masking\chi-038_3D_sh_echo-avg_desc-WM_mask.nii.gz"
sub08_shimmed_vert_levels_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\masking\register\chi-038_3D_sh_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
sub08_shimmed_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\fieldmap\masked_fieldmap\chi-038_3D_sh_desc-b0_fieldmap.nii.gz"
sub08_shimmed_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\sct_metrics\fieldmap_slicewise_metrics.csv"
sub08_shimmed_localfield_optpdf_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\bgfr\comp_bgfr\opt_pdf\chi-038_3D_sh_desc-localfield.nii.gz"
sub08_shimmed_chimap_closedform_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\dipole_inversion\optimized\closedForm\from_PDF_optimized\chi-038_3D_sh_desc-chimap.nii.gz"
sub08_shimmed_chimap_medi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\dipole_inversion\comp_di\medi\optimized\from_PDF_optimized\chi-038_3D_sh_desc-chimap.nii.gz"
sub08_shimmed_chimap_fansi_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\qsm_processing\3D\shim\sc-qsm-output\chi-038_3D_sh\dipole_inversion\comp_di\fansi\optimized\from_PDF_optimized\chi-038_3D_sh_desc-chimap.nii.gz"
# Optional prediction of SH+ACDC from the SH acquisition
sub08_baseline_predict_b0_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\dcms_and_niftis\dcms_and_mac_processed\rt_shim_nifti_3D\shim_coeff\fieldmap_calculated_shim.nii.gz"
sub08_baseline_predict_b0_csv_path = r"R:\soothsayer\sc_qsm\sc_qsm\neuropoly_data\chi_038\dcms_and_niftis\dcms_and_mac_processed\rt_shim_nifti_3D\shim_coeff\chi_038_st_predict_fieldmap_slicewise_metrics.csv"

# PARTICIPANT DICTIONARY — automatically built from the editable paths above
# Importing this module does not open images or CSVs.
PATH_FIELDS = (
    "magnitude_path", "sc_mask_path", "gm_mask_path", "wm_mask_path",
    "vert_levels_path", "b0_path", "b0_csv_path",
    "localfield_optpdf_path", "chimap_closedform_path",
    "chimap_medi_path", "chimap_fansi_path",
)

participants = {}
for number in range(1, 9):
    prefix = f"sub{number:02d}"
    participant_id = f"sub-{number:02d}"
    participants[participant_id] = {
        "sh": {
            field: globals()[f"{prefix}_baseline_{field}"]
            for field in PATH_FIELDS
        },
        "sh_acdc": {
            field: globals()[f"{prefix}_shimmed_{field}"]
            for field in PATH_FIELDS
        },
        "predicted_sh_acdc": {
            "b0_path": globals()[f"{prefix}_baseline_predict_b0_path"],
            "b0_csv_path": globals()[f"{prefix}_baseline_predict_b0_csv_path"],
            "b0_vert_csv_path": globals()[f"{prefix}_baseline_predict_b0_vert_csv_path"],
        },
    }

# Example access:
# participants["sub-06"]["sh_acdc"]["localfield_optpdf_path"]

