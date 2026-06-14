#!/usr/bin/env bash
#
# This function will generate dynamic shim coefficients based on a field map from a multi-echo gradient recalled echo (meGRE) scan
# This is done in real time with the subject in the scanner, the table position needs to be the same ...
# A total field map is calculated with ROMEO
# meGRE magnitude image is used to generate an ROI and dynamic shim coefficients
#

# Go inside folder with all dicoms
# Use dcm2niix manually to ensure using the selected version installed on the conda environment
conda activate rt_shim 
mkdir "rt_shim_nifti"

dcm2niix -z y -f "chi_015_%z_%p_%s_e%e" -o "rt_shim_nifti" "20260611.chi_015.2026.06.11_12_19_56_DST_1.3.12.2.1107.5.99.3" 

cd "rt_shim_nifti"
# Now we are inside the rt_shim_nifti folder with all the niftis
open .

# Open a new terminal window and start running the SC mask from mag e1
# Masking, use SCT with the first echo of the anat meGRE
mkdir "masking"
# Copy the name of the first echo of the mag meGRE and use it in the command below
sct_deepseg spinalcord -i "" -o "masking/chi_015_sc_msk.nii.gz" 
sct_maths -i "masking/chi_015_sc_msk.nii.gz" -shape disk -dilate 10 -dim 2 -o "masking/chi_015_sc_msk_dil.nii.gz" 

# Go to the concatenate.sh script, edit the parameters and run!

# Calculate fieldmap with ROMEO
# Need to define echoes, with 7 echoes its: [5, 10, 15, 20, 25, 30, 35] (in ms)
MAG_FN = "chi_015_3D_OLD_mgre_5TEs_C3_11_mag.nii.gz"
PH_FN = "chi_015_3D_OLD_mgre_5TEs_C3_12_ph.nii.gz"
SC_FN = "chi_015_custom_shim_sc_msk.nii.gz"


# -p for phase, -m for magnitude, -B to calculate a B0 map, -t for echo times, -o for outpath
# TEs: [6.93, 11.85, 16.85, 21.85, 26.85] # Remember ROMEO expects the echo times in milliseconds
# For PF (6/8), the TEs [3.46, 9.20, 14.94, 20.68, 26.42]
# Remember to add the: --phase-offset-correction bipolar ONLY if there is bipolar readout!
julia /Users/mclogar/ROMEO.jl/romeo.jl -p "chi_015_2D_5meGRE_custom_shim_PF_10_ph.nii.gz" -m "chi_015_2D_5meGRE_custom_shim_PF_9_mag.nii.gz" -B -t "[3.46, 9.20, 14.94, 20.68, 26.42]" -o "fm_tests/test2_mskd" 

# To create the shim coefficients we need to create a json file and add some stuff
nano "fm_tests/test1_unmskd/B0.json"
'''
Copy and paste the following content in the json file but before open the .json file of any echo from the mag or ph
REMEMBER: Change both the shim settings and the table position according to the json, and the digits of the Imaging frequency
{
    "Manufacturer": "Siemens",
    "ManufacturersModelName": "Prisma_fit",
    "DeviceSerialNumber": "167006",
    "ImagingFrequency": "123.248962",
	"ShimSetting": [
		586,
		-7466,
		-9331,
		-1349,
		74,
		1972,
		570,
		60	],
	"TablePosition": [
		0,
		0,
		-11	],
    "PatientPosition": "HFS",
    "SoftwareVersions": "syngo MR XA60"
}
'''

# Calculate new shim coefficients

st_b0shim dynamic --fmap "fm_tests/test1_unmskd/B0.nii" --target "chi_015_2D_5meGRE_no_custom_shim_PF_5_e1.nii.gz" --mask "masking/chi_015_sc_msk_dil.nii.gz" --scanner-coil-order "0,1" --optimizer-method "pseudo_inverse" --output-file-format-scanner "slicewise-hrd" --output "shim_coeffs" || exit

# After shimming
mag_after_shim = "chi_015_2D_5meGRE_custom_shim_PF_9_mag.nii.gz"
ph_after_shim = "chi_015_2D_5meGRE_custom_shim_PF_10_ph.nii.gz"

julia /Users/mclogar/ROMEO.jl/romeo.jl -p "chi_015_2D_5meGRE_custom_shim_PF_10_ph.nii.gz" -m "chi_015_2D_5meGRE_custom_shim_PF_9_mag.nii.gz" -B -t "[3.96, 9.70, 15.44, 21.18, 26.92]" -o "fm_tests/test1_unmskd" 
