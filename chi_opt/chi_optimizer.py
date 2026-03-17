import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import os
import PyNomad as nomad
import time

path_to_chi_to_fm_fft = r"C:\Users\Admin\Documents\msc_project\Image-processing-strategies\chi_opt\susceptibility-to-fieldmap-fft"
sys.path.append(path_to_chi_to_fm_fft)
from functions import compute_fieldmap
#from susceptibility_to_fieldmap_fft.functions import compute_fieldmap

global opt_file_fn

best_solution = float('inf')  # Initialize with infinity0
opt_file_path = r"E:\msc_data\chi_fitting\joint_opt\all_dubs"
opt_file_fn = r"E:\msc_data\chi_fitting\joint_opt\all_dubs/log_opt.txt"

if os.path.exists(opt_file_path) and len(os.listdir(opt_file_path)) > 0:
    print("Folder txt_file exists and is not empty. Please delete the folder or choose a different name.")
    sys.exit(1)
else:
    os.makedirs(opt_file_path, exist_ok=True)
    print("Experiment folder created!")

with open(opt_file_fn, 'w') as file:
    file.write("Joint optimization with NOMAD: \n")


def log_solution(counter, chi_trachea, chi_lungs, obj_val):
    global best_solution
    if obj_val <= best_solution:
        if obj_val == best_solution:
            print("Found a solution with the same objective value, but different parameters.")
            with open(opt_file_fn, 'a') as file:
                file.write(f" Iteration #{counter}, Chi trachea: {chi_trachea} & Chi Lung: {chi_lungs}. Obj value: {best_solution} \n")
            return 0

        best_solution = obj_val
        
        with open(opt_file_fn, 'a') as file:
            file.write(f" \n")
            print(f"New best solution: Iteration #{counter}, Chi trachea: {chi_trachea} & Chi Lung: {chi_lungs}. Obj value: {best_solution} ")
        return 1
    
    else:
        print("No improvement in objective value.")
        return 0
    

def f_simple_double_opt(x):

    global counter, best_solution
    counter += 1  

    print('$$$$$$$$$$$$$$$$$$$$$$$$$')
    print(f"Iteration #{counter}")
    # COnver the PyNomad Eval Point to list for subscriptions
    chi_trachea =  x.get_coord(0)
    chi_lungs = x.get_coord(1)

    print(f"Chi for trachea: {chi_trachea}")
    print(f"Chi for lungs: {chi_lungs}")
    #print(f"Current values of variables: x[0] = {x[0]}, x[1] = {x[1]} & x[2] = {x[2]}")
    
    # Assign variables 
    # Now we get only 1 value 
    #chi_internal_air = 0.27 # ppm in SI units 

    sim_chi_data32[ind_trachea32] = chi_trachea
    sim_chi_data32[ind_lungs32] = chi_lungs
    sim_chi_data33[ind_trachea33] = chi_trachea
    sim_chi_data33[ind_lungs33] = chi_lungs
    sim_chi_data35[ind_trachea35] = chi_trachea
    sim_chi_data35[ind_lungs35] = chi_lungs

    # Now we gotta compute the B0 distribution in [ppm] using compute_bz from repository

    sim_b0_32_ppm = compute_fieldmap.compute_bz(sim_chi_data32, image_resolution = image_res32)
    sim_b0_33_ppm = compute_fieldmap.compute_bz(sim_chi_data33, image_resolution = image_res33)
    sim_b0_35_ppm = compute_fieldmap.compute_bz(sim_chi_data35, image_resolution = image_res35)

    # bz will be in [ppm] we go to Hz now
    sim_b0_32_Hz = sim_b0_32_ppm * central_freq_db32
    sim_b0_33_Hz = sim_b0_33_ppm * central_freq_exp_db33
    sim_b0_35_Hz = sim_b0_35_ppm * central_freq_db35
    
    chi1_name = str(str(float(f"{chi_trachea:.3f}")))#.replace(".","_") # to take away the minus sign can use .strip("-")) at the end
    chi2_name = str(str(float(f"{chi_lungs:.3f}")))#.replace(".","_")
    
    ##########################################################################################

    # We won't be creating and saving the maps now because it will be resource intensive
    
    # Now before saving the new fieldmap, lets demodulate with the SC mask
    iter_demod_value32 = np.mean(sim_b0_32_Hz[dmod_sim_mask32 > 0])
    iter_demod_value33 = np.mean(sim_b0_33_Hz[dmod_sim_maks33 > 0])
    iter_demod_value35 = np.mean(sim_b0_35_Hz[dmod_sim_mask35 > 0])

    print(f"db0_032 dmod value for iteration #{counter} => {iter_demod_value32}")
    print(f"db0_033 dmod value for iteration #{counter} => {iter_demod_value33}")
    print(f"db0_035 dmod value for iteration #{counter} => {iter_demod_value35}")

    demod_iter_fm_32_Hz = sim_b0_32_Hz - iter_demod_value32
    demod_iter_fm_33_Hz = sim_b0_33_Hz - iter_demod_value33
    demod_iter_fm_35_Hz = sim_b0_35_Hz - iter_demod_value35


    # Now we extract metrics manually

    dmod_sim_fm_vert_values32 = []
    dmod_sim_fm_vert_values33 = []
    dmod_sim_fm_vert_values35 = []

    for v in vertebrae_subset:
        # For dub32
        label32 = vertebra_label_map[v]
        mask32 = (sim32metric_mask == label32)
        mean_val32 = np.mean(demod_iter_fm_32_Hz[mask32])
        dmod_sim_fm_vert_values32.append(mean_val32)
        # For dub33
        label33 = vertebra_label_map[v]
        mask33 = (sim33metric_mask == label33)
        mean_val33 = np.mean(demod_iter_fm_33_Hz[mask33])
        dmod_sim_fm_vert_values33.append(mean_val33)
        # For dub35
        label35 = vertebra_label_map[v]
        mask35 = (sim35_metric_mask == label35)
        mean_val35 = np.mean(demod_iter_fm_35_Hz[mask35])
        dmod_sim_fm_vert_values35.append(mean_val35)

    # After this for loop we have the demodulated simulated fm values for each vertebrae in the subset list C5 to T7
    # Now lets compute the metrics


    residuals = np.concatenate([
        db0_032_avg_dmod_metric_values - dmod_sim_fm_vert_values32,
        db0_033_avg_dmod_metric_values - dmod_sim_fm_vert_values33,
        db0_035_avg_dmod_metric_values - dmod_sim_fm_vert_values35
    ])

    joint_difference = np.linalg.norm(residuals)

    # Track history for convergence
    history.append(joint_difference)
    history_chi_trachea.append(chi_trachea)
    history_chi_lungs.append(chi_lungs)

    objective_value = joint_difference

    plot_sol = log_solution(counter, chi_trachea, chi_lungs, joint_difference)

    if plot_sol:

        plt.clf() # Clear the plot to only have 1 in the output
        
        plt.plot(vertebrae_levels_joint_opt, dmod_sim_fm_vert_values32, marker = 'o', linestyle = '-', label = f"Simulated B0 map - Participant 1", color = "#DF3B25E2")  
        plt.plot(vertebrae_levels_joint_opt, db0_032_avg_dmod_metric_values, marker = 'x', linestyle = '--', label = "Measured B0 map - Participant 1", color = "#DF3B25E2") 
        
        plt.plot(vertebrae_levels_joint_opt, dmod_sim_fm_vert_values33, marker = 'o', linestyle = '-', label = f"Simulated B0 map - Participant 2", color = "#3652D1")
        plt.plot(vertebrae_levels_joint_opt, db0_033_avg_dmod_metric_values, marker = 'x', linestyle = '--', label = "Measured B0 map - Participant 2", color = "#3652D1")

        plt.plot(vertebrae_levels_joint_opt, dmod_sim_fm_vert_values35, marker = 'o', linestyle = '-', label = f"Simulated B0 map - Participant 3", color = "#0BB627")  
        plt.plot(vertebrae_levels_joint_opt, db0_035_avg_dmod_metric_values, marker = 'x', linestyle = '--', label = "Measured B0 map - Participant 3", color = "#0BB627") 


        plt.title(f"Nomad-Optimized $\\chi$ Trachea= {chi1_name} [ppm], $\\chi$ Lung= {chi2_name} [ppm]")

        plt.xlabel('Vertebral Level', fontsize=16)
        plt.ylabel('B0 [Hz]', fontsize=16)
        plt.legend()

        yticks = range(-120, 120, 20) 
        plt.yticks(yticks)

        plt.grid(True, which='both', linestyle='--', linewidth=0.7)
        plt.tight_layout()
        
        plot_name = f"iter_{counter}_chi_traecha_{chi1_name}_chi_lungs_{chi2_name}.png"
        plot_path = os.path.join(opt_file_path, plot_name)
        # Show the plot
        plt.savefig(plot_path, dpi=300)
        plt.close()
    else:
        print("No improvement, not plotting.")

    
    rawBBO = str(objective_value)
    x.setBBO(rawBBO.encode("UTF-8"))
    return 1

##############################################################################################


home_path = r"C:\Users\Admin\Documents\msc_project\Image-processing-strategies\chi_opt"
run_number = "triple_participant_32_32_35_joint_opt"

# Everytime you run the code, it will create a new folder with the run number and restart the counter
path_to_iter_fms = r"E:\msc_data\chi_fitting\joint_opt\all_participants_fms\iter_fms"
path_to_iter_metrics = r"E:\msc_data\chi_fitting\joint_opt\all_participants_fms\iter_metrics"

path_to_iter_fms = os.path.join(path_to_iter_fms, run_number)
path_to_iter_metrics = os.path.join(path_to_iter_metrics,run_number)
counter = 0

#########
# This don't change, this are used with the simulation's FOV for each subject
path_to_sim32_metric_mask = r"E:/msc_data/chi_fitting/db0_032/fm/sim/D2_D3_masks/cord_mask_labeled.nii.gz"
path_to_dmod32_mask = r"E:/msc_data/chi_fitting/db0_032/fm/sim/D2_D3_masks/cord_mask.nii.gz"
path_to_chimap32 = r"E:/msc_data/chi_fitting/db0_032/fm/sim/B1_chi_maps/db0_032_sus_opt_map.nii.gz" # -4.36ppm for chi_trachea and chi_lungs as initial values
path_to_segs32 = r"E:/msc_data/chi_fitting/db0_032/fm/chi-opt2/grouped_wb_segs.nii.gz"

path_to_sim33_metric_mask = r"E:\msc_data/chi_fitting/dB0_033_dup1/fm/sim/D2_D3_masks/cord_mask_labeled.nii.gz"
path_to_dmod33_msk = r"E:\msc_data/chi_fitting/dB0_033_dup1/fm/sim/D2_D3_masks/cord_mask.nii.gz"
path_to_chimap33 = r"E:\msc_data/chi_fitting/dB0_033_dup1/fm/sim/B1_chi_maps/dB0_033_mod0.nii.gz" # -4.2ppm for chi_trachea and chi_lungs as initial values
path_to_segs33 = r"E:\msc_data/chi_fitting/dB0_033_dup1/fm/chi-opt/final_merged_wb_segs.nii.gz"

path_to_sim35_metric_mask = r"E:\msc_data/chi_fitting/dB0_035/fm/sim/D2_D3_masks/cord_mask_labeled.nii.gz"
path_to_dmod35_mask = r"E:\msc_data/chi_fitting/dB0_035/fm/sim/D2_D3_masks/cord_mask.nii.gz"
path_to_chimap35 = r"E:\msc_data/chi_fitting/dB0_035/fm/sim/B1_chi_maps/dB0_035_abs_chi_dist.nii.gz" # -4.2 for chi_trachea and chi_lungs as initial values
path_to_segs35 = r"E:\msc_data\chi_fitting\dB0_035\fm\chi-opt3\final_merged_wb_segs.nii.gz"

# In vivo data loading
db0_032_avg_dmod_metric_values_df = pd.read_csv(r"E:\msc_data\chi_fitting\db0_032\fm\chi-opt3\dB0_032_avg_dmod_meas_metrics_c5t8.csv")
db0_032_avg_dmod_metric_values = db0_032_avg_dmod_metric_values_df['WA()']
db0_032_avg_dmod_metric_values = db0_032_avg_dmod_metric_values[:-1]
db0_032_avg_dmod_metric_values_list = db0_032_avg_dmod_metric_values.tolist()

db0_033_avg_dmod_metric_values_df = pd.read_csv(r"E:\msc_data\chi_fitting\dB0_033_dup1\fm\chi-opt3\dB0_033_avg_dmod_meas_metrics_c5t7.csv")
db0_033_avg_dmod_metric_values = db0_033_avg_dmod_metric_values_df['WA()']
db0_033_avg_dmod_metric_values_list = db0_033_avg_dmod_metric_values.tolist()


db0_035_avg_dmod_metric_values_df = pd.read_csv(r"E:\msc_data\chi_fitting\dB0_035\fm\chi-opt3\dB0_035_avg_dmod_meas_metrics_c3t7.csv")
db0_035_avg_dmod_metric_values = db0_035_avg_dmod_metric_values_df['WA()']
db0_035_avg_dmod_metric_values = db0_035_avg_dmod_metric_values[2:] # Taking out the first two values to have c5 to t7
db0_035_avg_dmod_metric_values_list = db0_035_avg_dmod_metric_values.tolist()

gamma_bar = 42.58 # MHz/T
B0 = 3 # [T]

# If the central frequency from the scanner is 123.249489 MHz, it meas the B0 strenght we need to simulate is:
central_freq_db32 = 123.249521 # in MHz
B0_used_scan32 = central_freq_db32 /gamma_bar
print("The B0 to use in the db0_032 simulation should be: ", B0_used_scan32, "T")

central_freq_exp_db33 = 123.249489 # in MHz 
B0_used_scan33 = central_freq_exp_db33 /gamma_bar
print("The B0 to use in the simulation should be: ", B0_used_scan33, "T")

central_freq_db35 = 123.249391 # in MHz
B0_used_scan35 = central_freq_db35 /gamma_bar
print("The B0 to use in the db0_035 simulation should be: ", B0_used_scan35, "T")


history = [] 
history_chi_trachea = []
history_chi_lungs = []

# Loading dependencies outside obj. function to decrease computational needs

# Load the simulated susceptibility map in ppm
sim_chi_img35 = nib.load(path_to_chimap35)
sim_chi_data35 = sim_chi_img35.get_fdata()

sim_chi_img33 = nib.load(path_to_chimap33)
sim_chi_data33 = sim_chi_img33.get_fdata()

sim_chi_img32 = nib.load(path_to_chimap32)
sim_chi_data32 = sim_chi_img32.get_fdata()

# Load segmentation labels that create the chimaps
ROI_img35 = nib.load(path_to_segs35)
ROI_data35 = ROI_img35.get_fdata()

ROI_img33 = nib.load(path_to_segs33)
ROI_data33 = ROI_img33.get_fdata()

ROI_img32 = nib.load(path_to_segs32)
ROI_data32 = ROI_img32.get_fdata()


# Find indices with the labels we want to update
ind_trachea35 = np.where((ROI_data35 == 8))
ind_lungs35 = np.where((ROI_data35 == 7))

ind_lungs33 = np.where((ROI_data33 == 7))
ind_trachea33 = np.where((ROI_data33 == 8))

ind_trachea32 = np.where((ROI_data32 == 8))
ind_lungs32 = np.where((ROI_data32 == 7))

# Load the spinal cord masks for each dub
dmod_sim_mask35 = nib.load(path_to_dmod35_mask).get_fdata()
dmod_sim_maks33 = nib.load(path_to_dmod33_msk).get_fdata()
dmod_sim_mask32 = nib.load(path_to_dmod32_mask).get_fdata()

# Load the metric mask data for each dub
sim32metric_mask = nib.load(path_to_sim32_metric_mask).get_fdata()
sim33metric_mask = nib.load(path_to_sim33_metric_mask).get_fdata()
sim35_metric_mask = nib.load(path_to_sim35_metric_mask).get_fdata()

vertebra_label_map = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "T1": 8, "T2": 9, "T3": 10, "T4": 11, "T5": 12, "T6": 13, "T7": 14}
vertebrae_subset = ["C5", "C6", "C7", "T1", "T2", "T3", "T4", "T5", "T6", "T7"] # The subset of vertebrae we want to use for the optimization

# With that done we can plot them:
vertebrae_levels_joint_opt = ['C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'] # From 5 to 14

_, image_res32, _ = compute_fieldmap.load_sus_dist(path_to_chimap32)
_, image_res33, _ = compute_fieldmap.load_sus_dist(path_to_chimap33)
_, image_res35, _ = compute_fieldmap.load_sus_dist(path_to_chimap35)

# Using PyNomad for optimization# Using optimize minimize from scipy
# Set initial values, boundaries and run optimization
nomad_params = [
    "DIMENSION 2", 
    "BB_INPUT_TYPE (R R)",
    "BB_OUTPUT_TYPE OBJ",
    "MAX_BB_EVAL 200",
    "DISPLAY_DEGREE 2",
    "DISPLAY_ALL_EVAL false",
    "DISPLAY_STATS BBE OBJ",
    "VNS_MADS_SEARCH true", # Optional Variable Neighborhood Search
    "VNS_MADS_SEARCH_TRIGGER 0.75" # Max desired ration of VNS BBevals over the total number of BBevals
]
x0 = [0.27, -4.2] # 
# First bound is trachea // Depends on objective code !!!
# Second bound is Lung // Depends on objective code !!!
# Check the MD above!
lb = [-5, -5]
ub = [0.3, 0.2]

if counter != 0 :
        # This means that you forgot to change the folder run number, to avoid mixing tests, please run that cell 
        # Changing the number after run!
    print("Please change run # to avoid mixing result folders :)")
else:
    start_time = time.time()
    result = nomad.optimize(f_simple_double_opt, x0, lb, ub, nomad_params)
    fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
    output = "\n".join(fmt)
    print("\nNOMAD results \n" + output + " \n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Optimization complete in: {elapsed_time:.3f} seconds")