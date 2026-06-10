#!/usr/bin/env bash
#
# This function will generate dynamic shim coefficients based on a field map from a multi-echo gradient recalled echo (meGRE) scan
# This is done in real time with the subject in the scanner, the table position needs to be the same ...
# A total field map is calculated with ROMEO
# meGRE magnitude image is used to generate an ROI and dynamic shim coefficients
#

# Now need to concatenate the echoes of the anatomical meGRE scan to prepare for fieldmap calculation
# Copy the name of each echo in the quotes
DUB_NAME="chi_015"
# Now, look at the scanner and take the name of the sequence you want to concatenatem
# The meGRE sequences go into 2 sequences "n" for the mag and "n+1" for the phase
SEQ_NAME="2D_7meGRE_no_custom_shim"

# Define the directory after running dcm2niix
NIFTI_DIR="/Users/mclogar/msc_data/SYNGO_TRANSFER/rt_shim_nifti"

MAG_SERIES=$(ls ${NIFTI_DIR}/${DUB_NAME}_${SEQ_NAME}_*_e1.nii.gz 2>/dev/null | grep -v '_ph' | sed "s/.*${SEQ_NAME}_\([0-9]*\)_e1\.nii\.gz/\1/")

PH_SERIES=$(ls ${NIFTI_DIR}/${DUB_NAME}_${SEQ_NAME}_*_e1_ph.nii.gz 2>/dev/null | sed "s/.*${SEQ_NAME}_\([0-9]*\)_e1_ph\.nii\.gz/\1/")

for SER in $MAG_SERIES; do
    MAG_ECHOES=$(ls ${NIFTI_DIR}/${DUB_NAME}_${SEQ_NAME}_${SER}_e*.nii.gz 2>/dev/null | grep -v '_ph' | tr '\n' ' ')
    st_image concat $MAG_ECHOES -o "${NIFTI_DIR}/${DUB_NAME}_${SEQ_NAME}_${SER}_mag.nii.gz" --axis 3
done

for SER in $PH_SERIES; do
    PH_ECHOES=$(ls ${NIFTI_DIR}/${DUB_NAME}_${SEQ_NAME}_${SER}_e*_ph.nii.gz 2>/dev/null | tr '\n' ' ')
    st_image concat $PH_ECHOES -o "${NIFTI_DIR}/${DUB_NAME}_${SEQ_NAME}_${SER}_ph.nii.gz" --axis 3
done