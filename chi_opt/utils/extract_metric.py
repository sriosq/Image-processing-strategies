
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import pandas as pd

# This script automates metric extraction using spinal cord toolbox!
# This helps creating the folders when they are not existant and also cleans up the final jupyternotebook + modular baby!

# Important librabry!
import subprocess

def extract_metrics(path_to_fm, path_to_sc_seg, path_to_vert_levels, path_to_output, method, vert, perlevel="1"):
    """
    Function to extract metrics using a terminal command.

    Parameters:
    - path_to_demod_b0 (str): Path to the demodulated B0 file.
    - path_to_sc_seg (str): Path to the spinal cord segmentation file.
    - path_to_vert_levels (str): Path to the vertebral levels file.
    - path_to_output (str): Path to the output CSV file.
    - method (str): Method used in extraction (default is "wa").
    - vert (str): Vertebral levels to process (default is "2:14").
    - perlevel (str): Per-level processing flag (default is "1").
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(path_to_output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Construct the command
    command = [
        "sct_extract_metric",
        "-i", path_to_fm,
        "-f", path_to_sc_seg,
        "-method", method,
        "-vert", vert,
        "-vertfile", path_to_vert_levels,
        "-perlevel", perlevel,
        "-o", path_to_output
    ]

    # Run the command and capture output
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print("Error occurred:", result.stderr)
    else:
        print("Metrics extracted successfully:", result.stdout)

def extract_auto():
    pass

    return 1


# Example usage
#path_to_demod_b0 = "C:/Users/User/msc_project/Image-processing-strategies/compare_fieldmap/data/lung_fitting_project/manually_simulated/demodulated/fftshift_bug/demod_fftshift_ISMRM_img.nii.gz"
#path_to_sc_seg = "D:/UNF_data/2024_08_23/slicer_work/t1w_wholebody_sc_complete.nii.gz"
#path_to_vert_levels = "D:/UNF_data/2024_08_23/slicer_work/t1w_wholebody_label_vertebrae_final.nii.gz"
#path_to_output = "C:/Users/User/msc_project/Image-processing-strategies/compare_fieldmap/data/lung_fitting_project/manually_simulated/extract_metrics/test/simulated_db0_030_metrics.csv"

# Call the function
#extract_metrics(path_to_demod_b0, path_to_sc_seg, path_to_vert_levels, path_to_output, method='wa', vert="2:14")
