#!/usr/bin/env bash

set -euo pipefail

echo "
This script runs the entire experiment for a given subject.

Arguments:
1. Path to the DICOM directory
2. Subject name or tag
3. Base sequence name
4. Optional series number

Example:
rt_shim subset chi_test 3D_meGRE_no_custom_shim 5
"

SCRIPT_DIR=$(dirname "$(realpath "$0")")

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: rt_shim <DCM_DIR> <DUB_NAME> <SEQ_NAME> <ACQ_TYPE> [SERIES_NUMBER]"
    echo "  ACQ_TYPE must be 2D or 3D"
    exit 1
fi

DCM_DIR=$1
DUB_NAME=$2
SEQ_NAME=$3
ACQ_TYPE=$4 # Expect 2D or 3D
SELECTED_SERIES=${5:-}


# Check series being an integer first
if [[ -n "${SELECTED_SERIES}" && ! "${SELECTED_SERIES}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Series number must be an integer."
    exit 1
fi

# Coil profile directory hard-coded for now
COIL_PROFILES_DIR="/Users/mclogar/msc_data/SYNGO_TRANSFER/coil_profiles_july28_2026"
COIL_PATH="${COIL_PROFILES_DIR}/coil_profiles_NP15.nii.gz"
COIL_CONFIG_PATH="${COIL_PROFILES_DIR}/NP15_config.json"

# Output directories
NIFTI_DIR="${DCM_DIR}/rt_shim_nifti"
CONCAT_DIR="${NIFTI_DIR}/concat"
MASK_DIR="${NIFTI_DIR}/masking"
FMAP_DIR="${NIFTI_DIR}/fmap"
SHIM_COEFF_DIR="${NIFTI_DIR}/shim_coeff"


mkdir -p "${NIFTI_DIR}" "${CONCAT_DIR}" "${MASK_DIR}" "${FMAP_DIR}" "${SHIM_COEFF_DIR}" #-p to pass if already there

echo "Converting DICOMs to NIfTIs"

dcm2niix \
    -z y \
    -f "${DUB_NAME}_%z_%p_%s_e%e" \
    -o "${NIFTI_DIR}" \
    "${DCM_DIR}"

echo "Finding matching series"

ALL_SERIES=$(
    find "${NIFTI_DIR}" -maxdepth 1 -type f \
        -name "${DUB_NAME}_${SEQ_NAME}_*_e1.nii.gz" \
        ! -name "*_ph.nii.gz" |
    sed -E "s|.*/${DUB_NAME}_${SEQ_NAME}_([0-9]+)_e1\.nii\.gz|\1|" |
    sort -n |
    uniq
)

if [[ -z "${ALL_SERIES}" ]]; then
    echo "ERROR: No matching series found for ${SEQ_NAME}."
    exit 1
fi

echo "Available series:"
printf '%s\n' "${ALL_SERIES}"

if [[ -n "${SELECTED_SERIES}" ]]; then
    if ! printf '%s\n' "${ALL_SERIES}" | grep -qx "${SELECTED_SERIES}"; then
        echo "ERROR: Series ${SELECTED_SERIES} was not found."
        exit 1
    fi

    SER="${SELECTED_SERIES}"

elif [[ $(printf '%s\n' "${ALL_SERIES}" | wc -l | tr -d ' ') -eq 1 ]]; then
    SER="${ALL_SERIES}"

else
    echo "ERROR: Multiple matching series were found."
    echo
    echo "Run the command again with the desired series number:"
    echo "rt_shim \"${DCM_DIR}\" \"${DUB_NAME}\" \"${SEQ_NAME}\" <series_number>"
    exit 1
fi

MAG_SER="${SER}"
PH_SER=$((MAG_SER + 1))

MAG_FN="${CONCAT_DIR}/${DUB_NAME}_${SEQ_NAME}_${MAG_SER}_mag.nii.gz"
PH_FN="${CONCAT_DIR}/${DUB_NAME}_${SEQ_NAME}_${PH_SER}_ph.nii.gz"

MAG_ECHOES=$(
    find "${NIFTI_DIR}" -maxdepth 1 -type f \
        -name "${DUB_NAME}_${SEQ_NAME}_${MAG_SER}_e*.nii.gz" \
        ! -name "*_ph.nii.gz" |
    sort
)

PH_ECHOES=$(
    find "${NIFTI_DIR}" -maxdepth 1 -type f \
        -name "${DUB_NAME}_${SEQ_NAME}_${PH_SER}_e*_ph.nii.gz" |
    sort
)

if [[ -z "${MAG_ECHOES}" ]]; then
    echo "ERROR: No magnitude echoes found for series ${MAG_SER}."
    exit 1
fi

if [[ -z "${PH_ECHOES}" ]]; then
    echo "ERROR: No phase echoes found for series ${PH_SER}."
    exit 1
fi

echo "Selected acquisition:"
echo "  Magnitude series: ${MAG_SER}"
echo "  Phase series:     ${PH_SER}"

echo "Magnitude echoes:"
printf '%s\n' "${MAG_ECHOES}"

echo "Phase echoes:"
printf '%s\n' "${PH_ECHOES}"

st_image concat ${MAG_ECHOES} -o "${MAG_FN}" --axis 3
st_image concat ${PH_ECHOES} -o "${PH_FN}" --axis 3

TEs="[6.93, 11.85, 16.85, 21.85, 26.85]"

julia /Users/mclogar/ROMEO.jl/romeo.jl \
	-p "${PH_FN}" \
	-m "${MAG_FN}" \
	-B \
	-t "${TEs}" \
	-o "${FMAP_DIR}" 

mage1_json="${NIFTI_DIR}/${DUB_NAME}_${SEQ_NAME}_${MAG_SER}_e1.json"
cp "${mage1_json}" "${FMAP_DIR}/B0.json"

# A B0 will be created we can now:
gzip "${FMAP_DIR}/B0.nii"
FMAP="${FMAP_DIR}/B0.nii.gz" 

exit
# Because of the automation above, we can automatically run segmentation using e1 mag
# 
mage1="${NIFTI_DIR}/${DUB_NAME}_${SEQ_NAME}_${MAG_SER}_e1.nii.gz"

sc_msk_fn="${MASK_DIR}/${DUB_NAME}_${SEQ_NAME}_${MAG_SER}_sc_msk.nii.gz"
shim_msk_fn="${MASK_DIR}/${DUB_NAME}_${SEQ_NAME}_${MAG_SER}_dilated_sc_msk.nii.gz"

# CHeck if e1 exists:
if [[ ! -f "${mage1}" ]]; then
    echo "ERROR: Selected magnitude echo 1 does not exist:"
    echo "  ${mage1}"
    exit 1
fi

echo "
BALL
"

st_mask threshold \
    -i "${mage1}" \
    --thr 30 \
    -o "${shim_msk_fn}"

# Calculate new shim coefficients

if [[ "${ACQ_TYPE}" == "2D" ]]; then
echo "Using 2D Shimming dynamic shimming"

    st_b0shim dynamic \
        --fmap "${FMAP}" \
        --target "${mage1}" \
        --mask "${shim_msk_fn}" \
        --scanner-coil-order "0,1" \
        --optimizer-method "pseudo_inverse" \
        --output-file-format-scanner "slicewise-hrd" \
        --output "${SHIM_COEFF_DIR}"

elif [[ "${ACQ_TYPE}" == "3D" ]]; then
echo "Using 3D Shimming dynamic shimming"

    st_b0shim dynamic \
        --fmap "${FMAP}" \
        --target "${mage1}" \
		--slices volume \
        --mask "${shim_msk_fn}" \
        --coil "${COIL_PATH}" "${COIL_CONFIG_PATH}" \
        --optimizer-method "least_squares" \
        --output-file-format-coil "chronological-coil" \
        --output "${SHIM_COEFF_DIR}" 

fi

