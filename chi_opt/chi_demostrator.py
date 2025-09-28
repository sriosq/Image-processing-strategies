import nibabel as nib
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import sys
import pandas as pd
import os
import json


path_to_chi_to_fm_fft = r"C:\Users\Admin\Documents\msc_project\Image-processing-strategies\chi_opt\susceptibility-to-fieldmap-fft"
sys.path.append(path_to_chi_to_fm_fft)
#sys.path.append('/Users/evaalonsoortiz/Documents/python/susceptibility-to-fieldmap-fft/')
from functions import compute_fieldmap

#########
# This don't change, this are used with the simulation's FOV for each subject
path_to_sim32_metric_mask = "E:/msc_data/ismrm_2025/db0_032/fm/sim/D2_D3_masks/cord_mask_labeled.nii.gz"
path_to_dmod32_mask = "E:/msc_data/ismrm_2025/db0_032/fm/sim/D2_D3_masks/cord_mask.nii.gz"
path_to_chimap32 = "E:/msc_data/ismrm_2025/db0_032/fm/sim/B1_chi_maps/db0_032_sus_opt_map.nii.gz" # -4.2ppm internal air (past value)
path_to_segs32 = "E:/msc_data/ismrm_2025/db0_032/fm/chi-opt2/grouped_wb_segs.nii.gz"

path_to_sim35_metric_mask = "E:\msc_data/ismrm_2025/dB0_035/fm/sim/D2_D3_masks/cord_mask_labeled.nii.gz"
path_to_dmod35_mask = "E:\msc_data/ismrm_2025/dB0_035/fm/sim/D2_D3_masks/cord_mask.nii.gz"
path_to_chimap35 = "E:\msc_data/ismrm_2025/dB0_035/fm/sim/B1_chi_maps/dB0_035_abs_chi_dist.nii.gz" # -4.36 (past value)
path_to_segs35 = r"E:\msc_data\ismrm_2025\dB0_035\fm\chi-opt3\final_merged_wb_segs.nii.gz"

# load it:
db0_032_avg_dmod_metric_values_df = pd.read_csv(r"E:\msc_data\ismrm_2025\db0_032\fm\chi-opt3\dB0_032_avg_dmod_meas_metrics_c5t8.csv")
# crop to use c5 to t7 below

# 
db0_032_avg_dmod_metric_values = db0_032_avg_dmod_metric_values_df['WA()']
db0_032_avg_dmod_metric_values = db0_032_avg_dmod_metric_values[:-1] # Taking out the last value to have c5 to t7

# load it:
db0_035_avg_dmod_metric_values_df = pd.read_csv(r"E:\msc_data\ismrm_2025\dB0_035\fm\chi-opt3\dB0_035_avg_dmod_meas_metrics_c3t7.csv")

# Check the values:
db0_035_avg_dmod_metric_values = db0_035_avg_dmod_metric_values_df['WA()']
db0_035_avg_dmod_metric_values = db0_035_avg_dmod_metric_values[2:] # Taking out the first two values to have c5 to t7

central_freq_db32 = 123.249521 # in MHz
gamma_bar = 42.58 # MHz/T
# If the central frequency from the scanner is 123.249489 MHz, it meas the B0 strenght we need to simulate is:
B0_used_scan32 = central_freq_db32 /gamma_bar


central_freq_db35 = 123.249391 # in MHz
B0_used_scan35 = central_freq_db35 /gamma_bar

history = [] 
history_chi_trachea = []
history_chi_lungs = []

# Load the simulated susceptibility map in ppm
sim_chi_img35 = nib.load(path_to_chimap35)
sim_chi_data35 = sim_chi_img35.get_fdata()

# Load segmentation labels that create the chimaps
ROI_img35 = nib.load(path_to_segs35)
ROI_data35 = ROI_img35.get_fdata()

# Find indices with the labels we want to update
ind_trachea35 = np.where((ROI_data35 == 8))
ind_lungs35 = np.where((ROI_data35 == 7))

dmod_sim_mask35 = nib.load(path_to_dmod35_mask).get_fdata()

# Now for dub db0_032
# Load the simulated susceptibility map in ppm
sim_chi_img32 = nib.load(path_to_chimap32)
sim_chi_data32 = sim_chi_img32.get_fdata()

# Load segmentation labels that create the chimaps
ROI_img32 = nib.load(path_to_segs32)
ROI_data32 = ROI_img32.get_fdata()

# Find indices with the labels we want to update
ind_trachea32 = np.where((ROI_data32 == 8))
ind_lungs32 = np.where((ROI_data32 == 7))

dmod_sim_mask32 = nib.load(path_to_dmod32_mask).get_fdata()

# Load the metric mask data for each dub
sim32metric_mask = nib.load(path_to_sim32_metric_mask).get_fdata()
sim35_metric_mask = nib.load(path_to_sim35_metric_mask).get_fdata()

vertebra_label_map = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "T1": 8, "T2": 9, "T3": 10, "T4": 11, "T5": 12, "T6": 13, "T7": 14}
vertebrae_subset = ["C5", "C6", "C7", "T1", "T2", "T3", "T4", "T5", "T6", "T7"] # The subset of vertebrae we want to use for the optimization

# With that done we can plot them:
vertebrae_levels_joint_opt = ['C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'] # From 5 to 14


def set_susceptibility_and_show_B0(x1, x2):
    global sim_chi_data32, sim_chi_data35, ind_trachea32, ind_lungs32, ind_trachea35, ind_lungs35
    global dmod_sim_mask32, dmod_sim_mask35, sim32metric_mask, sim35_metric_mask
    global vertebrae_levels_joint_opt, vertebra_label_map
    global db0_032_avg_dmod_metric_values, db0_035_avg_dmod_metric_values
    global central_freq_db32, central_freq_db35, compute_fieldmap, path_to_chimap32, path_to_chimap35

    chi_trachea =  x1
    chi_lungs = x2

    sim_chi_data32[ind_trachea32] = chi_trachea
    sim_chi_data32[ind_lungs32] = chi_lungs
    sim_chi_data35[ind_trachea35] = chi_trachea
    sim_chi_data35[ind_lungs35] = chi_lungs

    # Now we gotta compute the B0 distribution in [ppm] using compute_bz from repository
    # Lets load the chi map with load_sus_dist from compute fieldmap to get the image res for FBFest
    chi_dist32, image_res32, affine_matrix32 = compute_fieldmap.load_sus_dist(path_to_chimap32)
    chi_dist35, image_res35, affine_matrix35 = compute_fieldmap.load_sus_dist(path_to_chimap35)

    sim_b0_32_ppm =  compute_fieldmap.compute_bz(sim_chi_data32, image_resolution = image_res32, buffer = 50) #, mode = "b0SimISMRM")
    sim_b0_35_ppm =  compute_fieldmap.compute_bz(sim_chi_data35, image_resolution = image_res35, buffer = 50) #, mode = "b0SimISMRM")
    # bz will be in [ppm] we go to Hz now
    sim_b0_32_Hz = sim_b0_32_ppm * central_freq_db32
    sim_b0_35_Hz = sim_b0_35_ppm * central_freq_db35
    
    chi1_name = str(str(float(f"{chi_trachea:.3f}")))#.replace(".","_") # to take away the minus sign can use .strip("-")) at the end
    chi2_name = str(str(float(f"{chi_lungs:.3f}")))#.replace(".","_")
    
    ##########################################################################################

    # We won't be creating and saving the maps now because it will be resource intensive
    
    # Now before saving the new fieldmap, lets demodulate with the SC mask
    iter_demod_value32 = np.mean(sim_b0_32_Hz[dmod_sim_mask32 == 1])
    iter_demod_value35 = np.mean(sim_b0_35_Hz[dmod_sim_mask35 == 1])

    demod_iter_fm_32_Hz = sim_b0_32_Hz - iter_demod_value32
    demod_iter_fm_35_Hz = sim_b0_35_Hz - iter_demod_value35

    # Now we extract metrics manually

    dmod_sim_fm_vert_values32 = []
    dmod_sim_fm_vert_values35 = []

    for v in vertebrae_subset:
        label32 = vertebra_label_map[v]
        mask32 = (sim32metric_mask == label32)
        mean_val32 = np.mean(demod_iter_fm_32_Hz[mask32])
        dmod_sim_fm_vert_values32.append(mean_val32)
        # Now for dub 35
        label35 = vertebra_label_map[v]
        mask35 = (sim35_metric_mask == label35)
        mean_val35 = np.mean(demod_iter_fm_35_Hz[mask35])
        dmod_sim_fm_vert_values35.append(mean_val35)


    plt.figure(figsize=(10, 6))
    plt.plot(vertebrae_levels_joint_opt, dmod_sim_fm_vert_values32, marker = 'o', linestyle = '-', label = f"Simulated Subj1", color = "#EC3838")  
    plt.plot(vertebrae_levels_joint_opt, db0_032_avg_dmod_metric_values, marker = 'x', linestyle = '--', label = "In-vivo meas Subj1", color = "#EC3838") 

    plt.plot(vertebrae_levels_joint_opt, dmod_sim_fm_vert_values35, marker = 'o', linestyle = '-', label = f"Simulated Subj2", color = "#12269B")  
    plt.plot(vertebrae_levels_joint_opt, db0_035_avg_dmod_metric_values, marker = 'x', linestyle = '--', label = "In-vivo meas Subj2", color = "#12269B") 


    plt.title(f"Initial $\\chi$: Trachea: {chi1_name} & Lung: {chi2_name}", fontsize=16)

    plt.xlabel('Vertebral Level', fontsize=18)
    plt.ylabel('B0 [Hz]', fontsize=18)
    plt.legend(fontsize=16)

    yticks = range(-160, 160, 20) 
    plt.yticks(yticks, fontsize=18)
    plt.xticks(fontsize=18)   

    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.tight_layout()

        # Show the plot
    plt.show()



set_susceptibility_and_show_B0(0, -7) # Example values

