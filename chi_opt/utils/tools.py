import nibabel as nib
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import sys
import pandas as pd
import os

def chi_to_fm_ppm(input_file, output_file):

    #Using a custom command for dB0 simulation

    command = [
    "compute_fieldmap",
    "-i", input_file,
    "-o", output_file,
    "-b", "50",
    "-m", "b0SimISMRM"
    ]

    # Run the command and capture output
    result = subprocess.run(command, capture_output=True, text=True)

    # We need to return the data from the fieldmap created

    
    # Check for errors
    if result.returncode != 0:
        print("Error occurred:", result.stderr)

    else:
        print("Fieldmap simulated correctly", result.stdout)
        sim_fm_img = nib.load(output_file)
        sim_fm_data = sim_fm_img.get_fdata()
        return sim_fm_data

#home_path = r"C:\Users\Admin\Documents\msc_project\Image-processing-strategies\chi_opt"
path_to_chi_to_fm_fft = r"C:\Users\Admin\Documents\msc_project\susceptibility-to-fieldmap-fft"
sys.path.append(path_to_chi_to_fm_fft)
from functions import compute_fieldmap

# Define this crucial function for a +1 ppm change in chi in a given ROI, returning a response vector for all vertebral levels inside the vertebral level map
def vertebral_response(
    roi_label,
    roi_data,
    image_res,
    central_freq,
    dmod_mask,
    metric_mask,
    vertebral_levels,
    vertebra_label_map,
    buffer,
    mode
):
    """Response to a +1 ppm susceptibility perturbation in one ROI."""

    unit_chi_map = (roi_data == roi_label).astype(float)

    response_hz = (
        compute_fieldmap.compute_bz(
            unit_chi_map,
            image_resolution=image_res,
            buffer = buffer,
            mode = mode
        )
        * central_freq
    )

    # Use the same demodulation as the optimization
    response_hz -= np.mean(response_hz[dmod_mask == 1])

    return np.array([
        np.mean(
            response_hz[
                metric_mask == vertebra_label_map[level]
            ]
        )
        for level in vertebral_levels
    ])

def extract_demodulated_sc_profile(
    chi_map,
    image_res,
    central_freq,
    dmod_mask,
    metric_mask,
    vertebral_levels,
    vertebra_label_map,
    buffer,
    mode
):
    field_hz = (
        compute_fieldmap.compute_bz(
            chi_map,
            image_resolution=image_res,
            buffer = buffer,
            mode = mode,
        )
        * central_freq
    )

    # Subtract the mean field across the complete SC
    sc_mean_hz = np.mean(
        field_hz[dmod_mask == 1]
    )

    demodulated_field_hz = (
        field_hz - sc_mean_hz
    )

    # Extract the mean field at each vertebral level
    sc_profile = np.array([
        np.mean(
            demodulated_field_hz[
                metric_mask
                == vertebra_label_map[level]
            ]
        )
        for level in vertebral_levels
    ])

    return sc_profile