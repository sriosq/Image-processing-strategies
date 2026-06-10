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

dcm2niix -z y -f "rt_shim_nifti/chi_015_%z_%p_%s_e%e" -o testing_from_vscode "SYNGO_TRANSFER/..." 

cd "rt_shim_nifti"
# Now we are inside the rt_shim_nifti folder with all the niftis
open .

# Now need to concatenate the echoes of the anatomical meGRE scan to prepare for fieldmap calculation
# Go to the concatenate.sh script, edit the parameters and run!

# Calculate fieldmap with ROMEO
# Need to define echoes, with 7 echoes its: [5, 10, 15, 20, 25, 30, 35] (in ms)
MAG_FN = "chi_015_2D_7meGRE_no_custom_shim_5_mag.nii.gz"
PH_FN = "chi_015_2D_7meGRE_no_custom_shim_6_ph.nii.gz"

# -p for phase, -m for magnitude, -B to calculate a B0 map, -t for echo times, -o for outpath
julia /Users/mclogar/ROMEO.jl/romeo.jl -p "chi_015_ph.nii.gz" -m "chi_015_mag.nii.gz" -B -t "[]" -o "fm_tests/test1_unmskd" 

# To create the shim coefficients we need to create a json file and add some stuff
nano "fm_tests/test1_unmskd/B0.json" || exit
'''
Copy and paste the following content in the json file but before open the .json file of any echo from the mag or ph
REMEMBER: Change both the shim settings and the table position according to the json, and the digits of the Imaging frequency
{
    "Manufacturer": "Siemens",
    "ManufacturersModelName": "Prisma_fit",
    "DeviceSerialNumber": "167006",
    "ImagingFrequency": "123.REPLACE_ME",
	"ShimSetting": [
		618,
		-7558,
		-8848,
		-493,
		233,
		1310,
		-629,
		416	],
	"TablePosition": [
		0,
		0,
		-58	],
    "PatientPosition": "HFS",
    "SoftwareVersions": "syngo MR XA60"
}
'''

# Masking, use SCT with the first echo of the anat meGRE
mkdir "masking" || exit
# Copy the name of the first echo of the mag meGRE and use it in the command below
sct_deepseg spinalcord -i "" -o "masking/chi_015_sc_msk.nii.gz" || exit
sct_maths -i "masking/chi_015_sc_msk.nii.gz" -shape disk -dilate 10 -dim 2 -o "masking/chi_015_sc_msk_dil.nii.gz" || exit

# Calculate new shim coefficients

st_b0shim dynamic --fmap "fm_tests/test1_unmskd/B0.nii" --target "chi_015_2D_7meGRE_no_custom_shim..." --mask "masking/chi_015_sc_msk_dil.nii.gz" --scanner-coil-order "0,1" --optimizer-method "pseudo_inverse" --output-file-format-scanner "slicewise-hrd" --output "shim_coeffs" || exit
