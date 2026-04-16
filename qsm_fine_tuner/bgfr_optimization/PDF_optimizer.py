#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
import nibabel as nib
import numpy as np
import sys
import json
import time
import pandas as pd
import matplotlib.pyplot as plt


def create_local_field(in1, in2, in3, in4 , output_basename, mask_filename, tol, num_iters, padSize):
    eng = matlab.engine.start_matlab()

    sepia_path = "R:/Poly_MSc_Code/libraries_and_toolboxes/sepia"
    xtra_tb_path = "R:/Poly_MSc_Code/libraries_and_toolboxes/toolboxes"

    eng.addpath(sepia_path)
    bfr_wrappers = eng.genpath("R:/Poly_MSc_Code/libraries_and_toolboxes/sepia/wrapper")
    eng.addpath(bfr_wrappers, nargout=0)

    all_funcs = eng.genpath("R:/Poly_MSc_Code/libraries_and_toolboxes/sepia")
    eng.addpath(all_funcs, nargout=0)

    path_to_MEDI_tb = "R:/Poly_MSc_Code/libraries_and_toolboxes/toolboxes/MEDI_toolbox"
    medi_sama = eng.genpath(path_to_MEDI_tb)
    eng.addpath(medi_sama, nargout = 0)

    #  PDF  Parameters
    tolerance = tol
    iterations = num_iters
    padSize = matlab.double(padSize)

    bfr_params = {
    
    'general' : {
        'isBET' : '0',
        'isInvert':'0',
        'isRefineBrainMask' : '0'
    },
    'bfr':{
    'method': "PDF",
    'tol': tolerance,
    'iteration': iterations,
    'padSize': padSize,
    "refine_method" : "None",
    "refine_order" : 4,
    'erode_radius': 0,
  'erode_before_radius': 0}

    }   

    eng.python_wrapper(in1, in2, in3, in4 , 'PDF', output_basename, mask_filename, bfr_params, nargout = 0)
    print("Local Field Created! Calculate metrics and update parameters!")


def configure_experiment_run(test_fn, first_line="Optimization results: "):
    global gm_mask_data, wm_mask_data, iter_folder, txt_file_path, results_vector
    results_vector = []
    gm_mask_img = nib.load(r"E:\msc_data\sc_qsm\final_gauss_sims/masks/sc_gm_crop.nii.gz")
    gm_mask_data = gm_mask_img.get_fdata()

    wm_mask_img = nib.load(r"E:\msc_data\sc_qsm\final_gauss_sims/masks/sc_wm_crop.nii.gz")
    wm_mask_data = wm_mask_img.get_fdata()

    print("GM and WM masks loaded successfully.")

    iter_folder = rf"E:\msc_data\sc_qsm\final_gauss_sims\feb_2026\bgfr_opt\snr_30\iter_PDF/{test_fn}"
    
    if os.path.exists(iter_folder) and len(os.listdir(iter_folder)) > 0:
        print("Folder already exists and is not empty. Please delete the folder or choose a different name.")
        sys.exit(1)
    else:
        os.makedirs(iter_folder, exist_ok=True)
        print("Experiment folder created!")

    txt_file_path = rf"E:\msc_data\sc_qsm\final_gauss_sims\feb_2026\bgfr_opt\snr_30\iter_PDF/{test_fn}.txt"
    with open(txt_file_path, 'w') as file:
        first_line_txt =  first_line + "\n"
        file.write(first_line_txt)
        
    print("Results file created at:", txt_file_path)

def load_groun_truth_data():
    global crop_gt_avg_sc_ref_swiss_crop_fm_Hz_data
    crop_gt_avg_sc_ref_swiss_crop_fm_Hz_data = nib.load(r"E:\msc_data\sc_qsm\final_gauss_sims\feb_2026\gt_data\bgfr_gt_ref_avg_onlySC_fm_Hz_crop.nii.gz").get_fdata()# This loads the Ground truth image with the Swiss Acq. Parameters FOV
    print("Ground truth local field loaded")

def log_best_solution(obj_value, iteration, tolerance, max_iters, padSize, gm_rmse, wm_rmse, wRMSE=None):
    global best_obj_value
    total_rmse = gm_rmse + wm_rmse

    if obj_value <= best_obj_value:
        out_text = f"Iter #{iteration}: Tolerance: {tolerance}, Max Iters: {max_iters}, PadSize: {padSize} || GM RMSE: {gm_rmse}, WM RMSE: {wm_rmse} | RMSE: {total_rmse} \n"
        out_text = out_text + "| wRMSE: " + str(wRMSE) + "\n" 
        if obj_value == best_obj_value:
            print("Found a solution with the same objective value, but different parameters.")
            with open(txt_file_path, 'a') as file:
                file.write(out_text)

        best_obj_value = obj_value
        print(f"New best solution found: {obj_value}")
        
        with open(txt_file_path, 'a') as file:
            file.write(out_text)

def pdf_optimizer(x):
    global counter

    #matrix_Size = [301, 351, 128]
    #voxelSize = [0.976562, 0.976562, 2.344]

    tolerance = 0.1
    num_iters = 250
    padSize = x.get_coord(0)
    
    iteration_fn = f"pdf_run{counter}/"

    output_fn = str(os.path.join(iter_folder,iteration_fn+"Sepia"))

    if not os.path.exists(output_fn):
        os.makedirs(output_fn)
        print("Created folder for new iteration #",counter)
    
    print("Output FN used:", output_fn)

    #custom_fm_path = str(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\fm_tests\test1_simple/B0.nii")
    custom_fm_path = str(r"E:\msc_data\sc_qsm\final_gauss_sims\November_2025\mrsim_outputs\custom_params_snr_30\fm_tests\test1_simple\B0.nii")
    # We can test using test1_simple or test2_msk_apply, the difference is that the second one has a mask applied and the first one does not
    custom_header_path = str(r"E:\msc_data\sc_qsm\final_gauss_sims\November_2025\mrsim_outputs\qsm_sc_phantom_custom_params.mat")
    mask_filename = str(r"E:\msc_data\sc_qsm\final_gauss_sims\masks/only_sc_crop.nii.gz")
    
    sepia_noise_sd = str(r"E:\msc_data\sc_qsm\final_gauss_sims\feb_2026\fm_tests\Sepia_noisesd.nii.gz")
    in1 = custom_fm_path
    in2 = ""
    in3 = sepia_noise_sd # The third input is for the noise SD map derived from the phase, calculated with SEPIA 
    # PDF by default creates a ones matrix multiply by 1e-4. 
    in4 = custom_header_path

    create_local_field(in1, in2, in3, in4, output_fn, mask_filename, tolerance, num_iters, padSize)
    # Import local field for RMSE calculation
    new_local_field_path = os.path.join(iter_folder,iteration_fn + "Sepia_localfield.nii.gz")
    
    print("Local field import from:", new_local_field_path)

    local_field_img = nib.load(new_local_field_path)
    local_field_data = local_field_img.get_fdata()

    # Now, we compute the difference between current local field with the Ground Truth
    pixel_wise_difference = crop_gt_avg_sc_ref_swiss_crop_fm_Hz_data - local_field_data
    gm_diff = pixel_wise_difference[gm_mask_data==1]
    wm_diff = pixel_wise_difference[wm_mask_data==1]

    gm_mean_diff = np.mean(gm_diff)
    gm_std_diff = np.std(gm_diff)
    gm_rmse = np.sqrt(np.mean(gm_diff ** 2))

    wm_mean_diff = np.mean(wm_diff)
    wm_std_diff = np.std(wm_diff)
    wm_rmse = np.sqrt(np.mean(wm_diff ** 2))
    
    print("########################")
    print("Metrics for Iteration #",counter)
    print("GM vs GT")
    print(f"  Mean difference: {gm_mean_diff:.5f}")
    print(f"  Std deviation: {gm_std_diff:.5f}")
    print(f"  RMSE: {gm_rmse:.5f}")

    print("WM vs GT")
    print(f"  Mean difference: {wm_mean_diff:.5f}")
    print(f"  Std deviation: {wm_std_diff:.5f}")
    print(f"  RMSE: {wm_rmse:.5f}")
    print("########################")

    # Compute loss of voxels in WM and GM
    tot_gm_voxels = np.sum(gm_mask_data == 1)
    tot_wm_voxels = np.sum(wm_mask_data == 1)
    tot_voxels = tot_gm_voxels + tot_wm_voxels
    
    valid_voxels = local_field_data != 0
    
    kept_gm_voxels = np.sum((gm_mask_data > 0) & valid_voxels)
    kept_wm_voxels = np.sum((wm_mask_data > 0) & valid_voxels)

    gm_retention = kept_gm_voxels / tot_gm_voxels # Fraction kept in GM  (0 to 1)
    wm_retention = kept_wm_voxels / tot_wm_voxels # Fraction kept in WM (0 to 1)

    retention_coeff = (kept_gm_voxels + kept_wm_voxels)/(tot_voxels) # Fraction 0 to 1 for total kept voxels

    objective_value = gm_rmse + wm_rmse

    wRMSE = objective_value / retention_coeff

    results_vector.append({
    "iter": counter,
    "tolerance": tolerance,
    "num_iters": num_iters,
    "padsize": padSize,
    "gm_rmse": gm_rmse,
    "wm_rmse": wm_rmse,
    "RMSE": objective_value,
    "wRMSE": wRMSE
})

    log_best_solution(objective_value, counter, tolerance, num_iters, padSize, gm_rmse, wm_rmse, wRMSE=wRMSE)

    print(f"Iter {counter}: Tolerance={tolerance}, Max # of iterations(PDF)={num_iters}, Zero padSize={padSize}, GM+WM RMSE={objective_value}, wRMSE:{wRMSE}")

    # Data to save
    sidecar_data = {
        'iteration': counter,
        'tolerance': float(tolerance),
        'Max # of iterations (PDF)': float(num_iters),
        'Zero padSize': float(padSize),
        'wm_RMSE': float(wm_rmse),
        'gm_RMSE': float(gm_rmse),
        'objective_value': float(objective_value),
        'wRMSE': float(wRMSE)
    }

    # Increase counter
    counter += 1

    # We want this to be saved in the precise run so:
    json_filename = os.path.join(iter_folder, iteration_fn, "sidecar_data.json")
    with open(json_filename, 'w') as json_file:
        json.dump(sidecar_data, json_file, indent=4)
    print("Sidecar data saved to:", json_filename)

    rawBBO = str(objective_value)
    x.setBBO(rawBBO.encode("UTF-8"))


    return 1

#############################################################################################################################################

nomad_params = [
    "DIMENSION 1",
    "BB_INPUT_TYPE (I)",
    "BB_OUTPUT_TYPE OBJ",
    "MAX_BB_EVAL 100",
    "DISPLAY_DEGREE 1",
    "DISPLAY_ALL_EVAL false",
    "DISPLAY_STATS BBE OBJ",
    "VNS_MADS_SEARCH true", # Optional Variable Neighborhood Search
    "VNS_MADS_SEARCH_TRIGGER 0.75" # Max desired ration of VNS BBevals over the total number of BBevals
]


# For PDF the x0 should be [tolerance, num_iters, padSize]
# Bounds for this parameters are not directly given by formulas or references
# We select based on the experience and complexity of field variations
# Begin:
start_time = time.time()
x0 = [40] # Recommended by SEPIA (for brain)

lb = [1]

ub=[100]

counter = 0

first_line = "BGFR opt SNR 30, Pad size opt, Tol: 0.1 (def) and # iter 250 with noise SD-> Displaying error_term for CG_TOL : \n"
configure_experiment_run("debug_tol/test1", first_line)
best_obj_value = float('inf')
load_groun_truth_data()

result = nomad.optimize(pdf_optimizer, x0, lb, ub, nomad_params)

fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
output = "\n".join(fmt)
print("\nNOMAD results \n" + output + " \n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Optimization complete in: {elapsed_time:.3f} seconds")

results_df = pd.DataFrame(results_vector)

plt.plot(results_df["num_iters"], results_df["wRMSE"], marker="o")
plt.xlabel("Iteration")
plt.ylabel("wRMSE")
plt.title("NOMAD search trajectory")
plt.grid(True)
plt.show()
