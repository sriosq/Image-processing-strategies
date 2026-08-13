import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import os
import PyNomad as nomad
import time

home_path = r"C:\Users\Admin\Documents\msc_project\Image-processing-strategies\chi_opt"
path_to_chi_to_fm_fft = r"C:\Users\Admin\Documents\msc_project\susceptibility-to-fieldmap-fft"#
# Another path: r"C:\Users\Admin\Documents\msc_project\Image-processing-strategies\chi_opt\susceptibility-to-fieldmap-fft"
sys.path.append(path_to_chi_to_fm_fft)
from functions import compute_fieldmap
#from susceptibility_to_fieldmap_fft.functions import compute_fieldmap

def log_solution(counter, chi_trachea, chi_lungs, obj_val):
    global best_solution

    if obj_val < best_solution:
        best_solution = obj_val

        message = (
            f"New best solution: Iteration #{counter}, "
            f"Chi trachea: {chi_trachea}, "
            f"Chi lungs: {chi_lungs}, "
            f"Objective value: {best_solution}\n"
        )

        print(message.strip())

        with open(opt_file_fn, "a", encoding="utf-8") as file:
            file.write(message)

        return 1

    elif np.isclose(obj_val, best_solution):
        message = (
            f"Equivalent solution: Iteration #{counter}, "
            f"Chi trachea: {chi_trachea}, "
            f"Chi lungs: {chi_lungs}, "
            f"Objective value: {obj_val}\n"
        )

        print(message.strip())

        with open(opt_file_fn, "a", encoding="utf-8") as file:
            file.write(message)

        return 0

    else:
        print("No improvement in objective value.")
        return 0
    
def save_plot(
    counter,
    chi_trachea,
    chi_lungs,
    vertebral_levels,
    avg_metrics_data,
    sim_metrics_data,
    output_directory
):
    os.makedirs(output_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        vertebral_levels,
        avg_metrics_data,
        marker="o",
        markersize=6,
        linestyle="--",
        linewidth=1.8,
        label="In-vivo average",
        color="#0072B2"
    )

    ax.plot(
        vertebral_levels,
        sim_metrics_data,
        marker="*",
        markersize=9,
        linestyle="-",
        linewidth=2.0,
        label="Simulation",
        color="#E69F00"
    )

    ax.set_title(
        f"In-vivo vs. Simulation Chi-018\n"
        f"Iteration {counter} | "
        f"χ trachea={chi_trachea:.3f}, "
        f"χ lungs={chi_lungs:.3f}"
    )
    ax.set_xlabel("Vertebral Level", fontsize=12)
    ax.set_ylabel("B0 [Hz]", fontsize=12)

    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.7)

    fig.tight_layout()

    figure_filename = (
        f"iter_{counter}_vertebral_comparison_"
        f"chi_trachea_{chi_trachea:.3f}_"
        f"chi_lungs_{chi_lungs:.3f}.png"
    )

    figure_path = os.path.join(
        output_directory,
        figure_filename
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved comparison plot: {figure_path}")


def f_nomad_opt(x):
    global counter, best_solution, dmod_sim_vert_values
    counter += 1
    print('$$$$$$$$$$$$$$$$$$$$$$$$$')
    print(f"Iteration #{counter}")
    # COnver the PyNomad Eval Point to list for subscriptions
    chi_trachea =  x.get_coord(0)
    chi_lungs = x.get_coord(1)

    print(f"Chi for trachea: {chi_trachea}")
    print(f"Chi for lungs: {chi_lungs}")

    # Step 1. Update chi value of trachea and lungs
    sim_chi_data[ind_trachea] = chi_trachea
    sim_chi_data[ind_lungs] = chi_lungs
    
    # Step 2. Compute the FM

    sim_b0_ppm = compute_fieldmap.compute_bz(sim_chi_data, image_resolution = image_res)
    sim_b0_Hz = sim_b0_ppm * central_freq_exp

    chi1_name = str(str(float(f"{chi_trachea:.3f}")))#.replace(".","_") # to take away the minus sign can use .strip("-")) at the end
    chi2_name = str(str(float(f"{chi_lungs:.3f}")))#.replace(".","_")

    # Step 3. demodulate and extract metrics
    dmod_value = np.mean(sim_b0_Hz[dmod_sim_mask == 1])

    print(f"Demodulation value for this iteration: {dmod_value}")
    dmod_sim_Hz = sim_b0_Hz - dmod_value

    # Now extract metrics manually instead of with subprocess to make it faster

    dmod_sim_vert_values = []

    for v in vertebrae_levels_opt:
        level = vertebra_label_map[v]
        mask = (metric_sim_mask==level)
        mean_value = np.mean(dmod_sim_Hz[mask])
        dmod_sim_vert_values.append(mean_value)

    # Step 4. Compute objective value and log solution
    difference = np.linalg.norm(np.array(dmod_sim_vert_values) - np.array(invivo_avg_metrics_values))
    print(f"Objective value for this iteration: {difference}")
    history.append(difference)
    history_chi_trachea.append(chi_trachea)
    history_chi_lungs.append(chi_lungs)

    plot_sol = log_solution(counter, chi_trachea, chi_lungs, difference)

    # Step 5. If the objective value is lower than the best solution, save the FM and chi maps
    if plot_sol == 1:
        # Save the FM and chi maps
        fm_filename = f"iter_{counter}_sim_b0_dmod_chi_trachea_{chi1_name}_chilung_{chi2_name}.nii.gz"
        chi_filename = f"iter_{counter}_sim_chi_chi_trachea_{chi1_name}_chi_lung_{chi2_name}.nii.gz"

        fm_path = os.path.join(path_to_iter_fms, fm_filename)
        chi_path = os.path.join(path_to_iter_fms, chi_filename)

        # Save the demodulated FM
        nib.save(nib.Nifti1Image(dmod_sim_Hz, affine_matrix), fm_path)
        # Save the updated chi map
        nib.save(nib.Nifti1Image(sim_chi_data, affine_matrix), chi_path)
            
        save_plot(
        counter=counter,
        chi_trachea=chi_trachea,
        chi_lungs=chi_lungs,
        vertebral_levels=vertebrae_levels_opt,
        avg_metrics_data=invivo_avg_metrics_values,
        sim_metrics_data=dmod_sim_vert_values,
        output_directory=path_to_iter_plots
    )

    else:
        print("No improvement, not saving.")

    
    rawBBO = str(difference)
    x.setBBO(rawBBO.encode("UTF-8"))
    return 1

##############################################################################################


home_path = r"C:\Users\Admin\Documents\msc_project\Image-processing-strategies\chi_opt"

best_solution = float('inf')  # Initialize with infinity0
run_number = "test1_no_extra_fm_creation_test"
path_to_iter_fms = r"Z:\neuropoly_data\chi_fitting\chi_018\chi_opt\iter_fms"
path_to_iter_metrics = r"Z:\neuropoly_data\chi_fitting\chi_018\chi_opt\iter_metrics"
global opt_file_fn
opt_file_fn = os.path.join(path_to_iter_metrics, "optimization_log.txt")
path_to_iter_plots = r"Z:\neuropoly_data\chi_fitting\chi_018\chi_opt\iter_plots"

counter = 0

#########
# This don't change, this are used with the simulation's FOV for each subject
path_to_sim_metric_mask = r"Z:\neuropoly_data\chi_fitting\chi_018\fm\sim\D2_D3_masks\t1w_wholebody_sc_msk_labeled.nii.gz" # Simulation
path_to_dmod_mask = r"Z:\neuropoly_data\chi_fitting\chi_018\fm\sim\D2_D3_masks\t1w_wholebody_sc_msk.nii.gz" # Simulation
path_to_chimap = r"Z:\neuropoly_data\chi_fitting\chi_018\fm\sim\B1_chi_maps\chi_018_chi_map.nii.gz" # -4.2 for both trachea and lungs as initial guess
path_to_segs = r"Z:\neuropoly_data\chi_fitting\chi_018\fm\sim\final_segmentations.nii.gz"

# In vivo data loading
# load in-vivo average metrics, this covers C3 to T8 (3 to 15)
invivo_avg_metrics = pd.read_csv(r"Z:\neuropoly_data\chi_fitting\chi_018\fm\C_dmod_meas\simple_avg_respiration.csv")
invivo_avg_metrics_values = invivo_avg_metrics['WA()']

gamma_bar = 42.58 # MHz/T
B0 = 3 # [T]

# Get info from json files of the scanner, the central frequency should be the same for both EXP and INSP
central_freq_exp = 123.24935 # in MHz  123.24935 vs 123.248944 (Exp vs Insp)
B0_used_scanner = central_freq_exp /gamma_bar
print("The B0 to use in the simulation should be: ", B0_used_scanner, "T")


history = [] 
history_chi_trachea = []
history_chi_lungs = []

# Loading dependencies outside obj. function to decrease computational needs
chi_dist, image_res, affine_matrix = compute_fieldmap.load_sus_dist(path_to_chimap)
sim_chi_img = nib.load(path_to_chimap)
sim_chi_data = sim_chi_img.get_fdata()

# Load segmentation labels that create the chimaps
ROI_img = nib.load(path_to_segs)
ROI_data = ROI_img.get_fdata()

# Find indices with the labels we want to update
ind_trachea = np.where((ROI_data == 113))
ind_lungs = np.where((ROI_data == 12))

dmod_sim_mask = nib.load(path_to_dmod_mask).get_fdata()
metric_sim_mask = nib.load(path_to_sim_metric_mask).get_fdata()

vertebra_label_map = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "T1": 8, "T2": 9, "T3": 10, "T4": 11, "T5": 12, "T6": 13, "T7": 14, "T8": 15}
vertebrae_levels_opt =  ['C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'] # From 3 to 15


# Using PyNomad for optimization
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
ub = [0.3, 5]

if counter != 0 :
        # This means that you forgot to change the folder run number, to avoid mixing tests, please run that cell 
        # Changing the number after run!
    print("Please change run # to avoid mixing result folders :)")
else:
    start_time = time.time()
    result = nomad.optimize(f_nomad_opt, x0, lb, ub, nomad_params)
    fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
    output = "\n".join(fmt)
    print("\nNOMAD results \n" + output + " \n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Optimization complete in: {elapsed_time:.3f} seconds")