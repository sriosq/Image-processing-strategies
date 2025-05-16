#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
import nibabel as nib
import numpy as np
import sys
import json
import time

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
    padSize = padSize # Zero padSize 

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


def configure_experiment_run(test_fn):
    global gm_mask_data, wm_mask_data, iter_folder, txt_file_path
    gm_mask_img = nib.load(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus/gm_mask_crop.nii.gz")
    gm_mask_data = gm_mask_img.get_fdata()

    wm_mask_img = nib.load(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus/wm_mask_crop.nii.gz")
    wm_mask_data = wm_mask_img.get_fdata()

    print("GM and WM masks loaded successfully.")

    iter_folder = rf"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\bgfr_opt\iter_PDF/{test_fn}"
    
    if os.path.exists(iter_folder) and len(os.listdir(iter_folder)) > 0:
        print("Folder already exists and is not empty. Please delete the folder or choose a different name.")
        sys.exit(1)
    else:
        os.makedirs(iter_folder, exist_ok=True)
        print("Experiment folder created!")

    txt_file_path = rf"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\bgfr_opt\iter_PDF/{test_fn}.txt"
    with open(txt_file_path, 'w') as file:
        file.write("Optimization results.\n")
        
    print("Results file created at:", txt_file_path)

def load_groun_truth_data():
    global wb_gt_avg_sc_ref_swiss_crop_fm_Hz_data
    wb_gt_avg_sc_ref_swiss_crop_fm_Hz_data = nib.load(r"E:\msc_data\sc_qsm\new_gauss_sims\gt_ref_avg_sc\gt_gauss_lf_Hz_swiss_crop.nii.gz").get_fdata()# This loads the Ground truth image with the Swiss Acq. Parameters FOV
    print("Ground truth local field loaded")

def log_best_solution(obj_value, iteration, tolerance, max_iters, padSize, gm_rmse, wm_rmse):
    global best_obj_value
    total_rmse = gm_rmse + wm_rmse
    if obj_value <= best_obj_value:
        if obj_value == best_obj_value:
            print("Found a solution with the same objective value, but different parameters.")
            with open(txt_file_path, 'a') as file:
                file.write(f"Iteration: {iteration}: OBJ {obj_value} // Tolerance: {tolerance}, Max Iters: {max_iters}, PadSize: {padSize}, GM RMSE: {gm_rmse}, WM RMSE: {wm_rmse} | RMSE: {total_rmse} \n")

        best_obj_value = obj_value
        print(f"New best solution found: {obj_value}")
        
        with open(txt_file_path, 'a') as file:
            file.write(f"Iteration: {iteration}: OBJ {obj_value} // Tolerance: {tolerance}, Max Iters: {max_iters}, PadSize: {padSize}, GM RMSE: {gm_rmse}, WM RMSE: {wm_rmse} | RMSE: {total_rmse} \n")

def pdf_optimizer(x):
    global counter

    #matrix_Size = [301, 351, 128]
    #voxelSize = [0.976562, 0.976562, 2.344]

    tolerance = x.get_coord(0)
    num_iters = x.get_coord(1)
    padSize = x.get_coord(2)
    
    iteration_fn = f"pdf_run{counter}/"

    output_fn = str(os.path.join(iter_folder,iteration_fn+"Sepia"))

    if not os.path.exists(output_fn):
        os.makedirs(output_fn)
        print("Created folder for new iteration #",counter)
    
    print("Output FN used:", output_fn)

    #custom_fm_path = str(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\fm_tests\test1_simple/B0.nii")
    custom_fm_path = str(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\fm_tests\test2_msk_apply/B0.nii")
    # We can test using test1_simple or test2_msk_apply, the difference is that the second one has a mask applied and the first one does not
    custom_header_path = str(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal/custom_qsm_sim.mat")
    mask_filename = str(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus/cord_mask_crop.nii.gz")

    in1 = custom_fm_path
    in2 = ""
    in3 = ""
    in4 = custom_header_path

    create_local_field(in1, in2, in3, in4, output_fn, mask_filename, tolerance, num_iters, padSize)
    # Import local field for RMSE calculation
    new_local_field_path = os.path.join(iter_folder,iteration_fn + "Sepia_localfield.nii.gz")
    
    print("Local field import from:", new_local_field_path)

    local_field_img = nib.load(new_local_field_path)
    local_field_data = local_field_img.get_fdata()

    # Now, we compute the difference between current local field with the Ground Truth
    pixel_wise_difference = wb_gt_avg_sc_ref_swiss_crop_fm_Hz_data - local_field_data
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

    # Compute mean fields within masks
    #gm_mean = np.mean(local_field_data[gm_mask_data == 1])
    #print("GM_mean: ", gm_mean)
    #wm_mean = np.mean(local_field_data[wm_mask_data == 1])
    #print("WM_mean: ", wm_mean)

    # Increase counter
    counter += 1

    # Objective: Maximize the difference between GM and WM means
    # PyNomad minimizes, so return negative to maximize
    objective_value = gm_rmse + wm_rmse
    log_best_solution(objective_value, counter, tolerance, num_iters, padSize, gm_rmse, wm_rmse)

    print(f"Iter {counter}: Tolerance={tolerance}, Max # of iterations(PDF)={num_iters}, Zero padSize={padSize}, GM+WM RMSE={objective_value}")

    # Data to save
    sidecar_data = {
        'iteration': counter,
        'tolerance': float(tolerance),
        'Max # of iterations (PDF)': float(num_iters),
        'Zero padSize': float(padSize),
        'wm_RMSE': float(wm_rmse),
        'gm_RMSE': float(gm_rmse),
        'objective_value': float(objective_value)
    }
    # We want this to be saved in the precie run so:
    json_filename = os.path.join(iter_folder, iteration_fn, "sidecar_data.json")
    with open(json_filename, 'w') as json_file:
        json.dump(sidecar_data, json_file, indent=4)
    print("Sidecar data saved to:", json_filename)

    rawBBO = str(objective_value)
    x.setBBO(rawBBO.encode("UTF-8"))

    return 1

#############################################################################################################################################

nomad_params = [
    "DIMENSION 3",
    "BB_INPUT_TYPE (R I I)",
    "BB_OUTPUT_TYPE OBJ",
    "MAX_BB_EVAL 600",
    "DISPLAY_DEGREE 2",
    "DISPLAY_ALL_EVAL false",
    "DISPLAY_STATS BBE OBJ"
]

# For PDF the x0 should be [tolerance, num_iters, padSize]
# Bounds for this parameters are not directly given by formulas or references
# We select based on the experience and complexity of field variations
# Begin:
start_time = time.time()
x0 = [0.1, 50, 40] # Recommended by SEPIA (for brain)

lb = [0.000001, 10, 20]

ub=[0.1, 100, 80]

counter = 0

configure_experiment_run("RMSE_test2_mskd_fm_integer_vars_600_evals")
best_obj_value = float('inf')
load_groun_truth_data()

result = nomad.optimize(pdf_optimizer, x0, lb, ub, nomad_params)

fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
output = "\n".join(fmt)
print("\nNOMAD results \n" + output + " \n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Optimization complete in: {elapsed_time:.3f} seconds")
