import nibabel as nib
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import sys
import pandas as pd
import os

path_to_chi_to_fm_fft = r"C:\Users\Admin\Documents\msc_project\Image-processing-strategies\chi_opt\susceptibility-to-fieldmap-fft"
sys.path.append(path_to_chi_to_fm_fft)
#sys.path.append('/Users/evaalonsoortiz/Documents/python/susceptibility-to-fieldmap-fft/')
from functions import compute_fieldmap

def check_single_chi_pair(path_to_chimap, path_to_segs, path_to_sim_dmod_msk, chi_trachea, chi_lungs, path_to_meas_data, meas_vert_levels, path_to_metric_msk, title, reverse_flag = 0, central_freq = 127.74):
    """
    Function to check and plot selected chi values
    
    Parameters:
    - path_to_chimap (str): Path to the chi map file.
    - path_to_segs (str): Path to the segmentation masks file.
    - chi_trachea (float): Expected value for trachea chi. Should be ROI = 8 in segmentation file
    - chi_lungs (float): Expected value for lungs chi. Should be ROI = 7 in segmentation file
    
    Returns:
    - 
    """
    
    # Load the chi map and segmentation masks
    chi_map = nib.load(path_to_chimap).get_fdata()
    segs = nib.load(path_to_segs).get_fdata()
    
    # Get the indexes of the trachea and lungs in the segmentation masks
    ind_trachea = np.where((segs == 8))
    ind_lungs = np.where((segs == 7))

    # Place the value of the chi map in the trachea and lungs
    chi_map[ind_trachea] = chi_trachea
    chi_map[ind_lungs] = chi_lungs
    
    chi_dist, image_res, affine_matrix = compute_fieldmap.load_sus_dist(path_to_chimap)
    sim_b0_ppm =  compute_fieldmap.compute_bz(chi_map, image_resolution = image_res, buffer = 50) #, mode = "b0SimISMRM")

    # Now, convert to Hz
    if central_freq != 127.74:
        print("Central frequency is not 127.74 MHz, converting to Hz with custom central frequency!")
        sim_b0_Hz = sim_b0_ppm * (central_freq)
    else:
        sim_b0_Hz = sim_b0_ppm * 127.74
    # To finish with the simulation work, we need to demodulate the simulated B0 fieldmap
    dmod_msk_data = nib.load(path_to_sim_dmod_msk).get_fdata()
    dmod_value = np.mean(sim_b0_Hz[dmod_msk_data == 1])
    dmod_sim_b0_Hz = sim_b0_Hz - dmod_value
    
    meas_data_df = pd.read_csv(path_to_meas_data)
    if reverse_flag:
        meas_data = meas_data_df["WA()"][::-1]  # Reverse the order needed
    else:
        meas_data = meas_data_df["WA()"]  # Reverse the order needed

    # Calculate average in simulation with mask
    metric_mask_data = nib.load(path_to_metric_msk).get_fdata()
    levels = np.unique(metric_mask_data)
    levels = levels[ levels != 0]  # Remove zero level, i.e. background
    avg_sim_values = []
    
    # Now to plot only in the same range as the measured
    min_meas_level = meas_vert_levels[0]
    max_meas_level = meas_vert_levels[-1]

    sim_levels = np.array(levels)
    sim_levels = sim_levels[(sim_levels >= min_meas_level) & (sim_levels <= max_meas_level)]
    
    # Now calculate only in this intersection
    for level in sim_levels:
        mask = metric_mask_data == level
        avg_value = np.mean(dmod_sim_b0_Hz[mask])
        avg_sim_values.append(avg_value)

    plt.plot(sim_levels, avg_sim_values, marker = '*', linestyle = '--', label = r"Simulated abs $\chi_{dist}$", color = "#16D8E6")
    plt.plot(meas_vert_levels, meas_data, marker = 'o', linestyle = '-', label = "Input measured data", color = "#F00C6B")
    

    plt.title(title)
    plt.xlabel('Vertebral Level', fontsize=12)
    plt.ylabel('B0 [Hz]', fontsize=12)
    plt.legend()
    # Customize grid and display
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.tight_layout()

    #yticks = range(-450, 450, 100) 
    #plt.yticks(yticks)

    # Show the plot
    plt.show()
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
chi_trachea = 0.2
chi_lungs = -4.2

meas_vert_levels_32 = ['C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'] # From 5 to 15
meas_vert_levels_33 = ['C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'] # From 5 to 15
meas_vert_levels_35 = list(range(3,15,1))


path_to_sim_metric_mask35 = r"E:\msc_data/ismrm_2025/dB0_035/fm/sim/D2_D3_masks/cord_mask_labeled.nii.gz"
path_to_dmod_mask35 = r"E:\msc_data/ismrm_2025/dB0_035/fm/sim/D2_D3_masks/cord_mask.nii.gz"
path_to_chimap35 = r"E:\msc_data/ismrm_2025/dB0_035/fm/sim/B1_chi_maps/dB0_035_abs_chi_dist.nii.gz" # -4.36 (past value)
path_to_segs35 = r"E:\msc_data\ismrm_2025\dB0_035\fm\chi-opt3\final_merged_wb_segs.nii.gz"

path_to_dmod_meas35 = r"E:\msc_data\ismrm_2025\dB0_035\fm\chi-opt3\dB0_035_avg_dmod_meas_metrics_c3t7.csv"
reverse_flag35 = 0  # This data is already reversed, so we don't need to reverse it when loading in the chi tester function

title35 = rf"dB0_035: $\chi \: trachea$ = {chi_trachea} ppm,  $\chi \: lungs$ = {chi_lungs} ppm"
check_single_chi_pair(path_to_chimap35, path_to_segs35, path_to_dmod_mask35, chi_trachea, chi_lungs, path_to_dmod_meas35, meas_vert_levels_35, path_to_sim_metric_mask35, title35, reverse_flag35, central_freq = 123.249391)

