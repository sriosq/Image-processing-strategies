#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
import nibabel as nib
import numpy as np
import sys
import json
import time

def create_local_field(in1, in2, in3, in4 , output_basename, mask_filename, max_radii, min_radii):
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

    #  VSHARP parameters
    max_radii = int(np.round(max_radii))
    min_radii = int(np.round(min_radii)-1)# We need to substract 1 to include the min_radii in the list
    
    radius_list = list(range(max_radii,min_radii,-1))
    radius_matlab = matlab.double(radius_list)

    bfr_params = {
    
    'general' : {
        'isBET' : '0',
        'isInvert':'0',
        'isRefineBrainMask' : '0'
    },
    'bfr':{
    'method': "VSHARP",
    "refine_method" : "None",
    "refine_order" : 4,
    'erode_radius': 0,
    'erode_before_radius': 0,
    'radius':radius_matlab}
    }   

    eng.python_wrapper(in1, in2, in3, in4 , 'VSHARP', output_basename, mask_filename, bfr_params, nargout = 0)
    print("Local Field Created! Calculate metrics and update parameters!")


def configure_experiment_run(test_fn):
    global gm_mask_data, wm_mask_data, iter_folder, txt_file_path
    gm_mask_img = nib.load(r"E:\msc_data\sc_qsm\final_gauss_sims\July_2025\mrsim_outputs\masks/sc_gm_crop.nii.gz")
    gm_mask_data = gm_mask_img.get_fdata()

    wm_mask_img = nib.load(r"E:\msc_data\sc_qsm\final_gauss_sims\July_2025\mrsim_outputs/masks/sc_wm_crop.nii.gz")
    wm_mask_data = wm_mask_img.get_fdata()

    print("GM and WM masks loaded successfully.")

    iter_folder = rf"E:\msc_data\sc_qsm\final_gauss_sims\July_2025\mrsim_outputs\custom_params/bgfr_opt\iter_VSHARP/{test_fn}"
   
    if os.path.exists(iter_folder) and len(os.listdir(iter_folder)) > 0:
        print("Folder already exists and is not empty. Please delete the folder or choose a different name.")
        sys.exit(1)
    else:
        os.makedirs(iter_folder, exist_ok=True)
        print("Experiment folder created!")

    txt_file_path = rf"E:\msc_data\sc_qsm\final_gauss_sims\July_2025\mrsim_outputs\custom_params/bgfr_opt\iter_VSHARP/{test_fn}.txt"
    with open(txt_file_path, 'w') as file:
        file.write("Optimization results.\n")
        
    print("Results file created at:", txt_file_path)
  
    

def load_groun_truth_data():
    global crop_gt_avg_sc_ref_swiss_crop_fm_Hz_data
    crop_gt_avg_sc_ref_swiss_crop_fm_Hz_data = nib.load(r"E:\msc_data\sc_qsm\final_gauss_sims\July_2025\ground_truth_data\bgfr_gt_ref_avg_sc_lf_Hz_crop.nii.gz").get_fdata()# This loads the Ground truth image with the Swiss Acq. Parameters FOV
    print("Ground truth local field loaded")

def log_best_solution(obj_value, iteration, max_radii, min_radii, gm_rmse, wm_rmse):
    global best_obj_value
    total_rmse = gm_rmse + wm_rmse
    if obj_value <= best_obj_value:
        if obj_value == best_obj_value:
            print("Found a solution with the same objective value, but different parameters.")
            with open(txt_file_path, 'a') as file:
                file.write(f"Iteration: {iteration}: OBJ {obj_value} // Max radii: {max_radii}, Min radii: {min_radii}, GM RMSE: {gm_rmse}, WM RMSE: {wm_rmse} | RMSE: {total_rmse} \n")

        best_obj_value = obj_value
        
        print(f"New best solution found: {obj_value}")
        
        with open(txt_file_path, 'a') as file:
            file.write(f"Iteration: {iteration}: OBJ {obj_value} // Max radii: {max_radii}, Min radii: {min_radii}, GM RMSE: {gm_rmse}, WM RMSE: {wm_rmse} | RMSE: {total_rmse} \n")

def vsharp_optimizer(x):
    global counter

    #matrix_Size = [301, 351, 128]
    #voxelSize = [0.976562, 0.976562, 2.344]

    max_radii = x.get_coord(0)
    min_radii = x.get_coord(1)
    
    iteration_fn = f"vsharp_run{counter}/"

    output_fn = str(os.path.join(iter_folder,iteration_fn+"Sepia"))

    if not os.path.exists(output_fn):
        os.makedirs(output_fn)
        print("Created folder for new iteration #",counter)
    
    print("Output FN used:", output_fn)

    #custom_fm_path = str(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\fm_tests\test1_simple/B0.nii")
    custom_fm_path = str(r"E:\msc_data\sc_qsm\final_gauss_sims\July_2025\mrsim_outputs/custom_params\fm_tests\test1_simple\B0.nii")
    # We can test using test1_simple or test2_msk_apply, the difference is that the second one has a mask applied and the first one does not
    custom_header_path = str(r"E:\msc_data\sc_qsm\final_gauss_sims\July_2025\mrsim_outputs/custom_params\qsm_sc_phantom_custom_params.mat")
    mask_filename = str(r"E:\msc_data\sc_qsm\final_gauss_sims\July_2025\mrsim_outputs\masks\qsm_processing_msk_crop.nii.gz")

    
    in1 = custom_fm_path
    in2 = ""
    in3 = ""
    in4 = custom_header_path

    create_local_field(in1, in2, in3, in4, output_fn, mask_filename, max_radii, min_radii)
    # Import local field for RMSE calculation
    new_local_field_path = os.path.join(iter_folder, iteration_fn + "Sepia_localfield.nii.gz")
    
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
    # Try to log the best solution
    log_best_solution(objective_value, counter, max_radii, min_radii, gm_rmse, wm_rmse)

    print(f"Iter {counter}: Max radii={max_radii}, Min radii ={min_radii}, GM+WM RMSE={objective_value}")

    # Data to save
    sidecar_data = {
        'iteration': counter,
        'max_radii': float(max_radii),
        'min_radii': float(min_radii),
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
    "DIMENSION 2",
    "BB_INPUT_TYPE (I I)",
    "BB_OUTPUT_TYPE OBJ",
    "MAX_BB_EVAL 200",
    "DISPLAY_DEGREE 2",
    #"STATS_FILE nomad_stats_test2_vsharp.txt $ BBE $ ( SOL )  & $ %.5EOBJ $ ( TIME )",
    "DISPLAY_ALL_EVAL false",
    "DISPLAY_STATS BBE OBJ",
    "VNS_MADS_SEARCH true", # Optional Variable Neighborhood Search
    "VNS_MADS_SEARCH_TRIGGER 0.75" # Max desired ration of VNS BBevals over the total number of BBevals
]
# For VSHARP the x0 should be [max_radii, min_radii]
# With our image, the largest axis is 343 mm and the smallest is 300 mm
# To ensure the largest kernel fits withing the image, upper bound for a should be smallest image dimension/2
# In this case 300/2 = 150
start_time = time.time()
x0 = [10, 3] # Recommended by SEPIA (for brain)
# I think the best would be to start with the largest kernel and then decrease it
#x0_largest = [150, 1] # This is the radii list

lb = [2, 1]

ub=[150, 149]

counter = 0

configure_experiment_run("RMSE_test1_200")
best_obj_value = float('inf')
load_groun_truth_data()

result = nomad.optimize(vsharp_optimizer, x0, lb, ub, nomad_params)

fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
output = "\n".join(fmt)
print("\nNOMAD results \n" + output + " \n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Optimization complete in: {elapsed_time:.3f} seconds")
