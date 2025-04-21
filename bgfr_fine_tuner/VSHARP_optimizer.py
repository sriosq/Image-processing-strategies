#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
import nibabel as nib
import numpy as np
import sys
import json

def create_local_field(in1, in2, in3, in4 , output_basename, mask_filename, max_radii, min_radii):
    eng = matlab.engine.start_matlab()

    sepia_path = "D:/Poly_MSc_Code/libraries_and_toolboxes/sepia"
    xtra_tb_path = "D:/Poly_MSc_Code/libraries_and_toolboxes/toolboxes"

    eng.addpath(sepia_path)
    bfr_wrappers = eng.genpath("D:/Poly_MSc_Code/libraries_and_toolboxes/sepia/wrapper")
    eng.addpath(bfr_wrappers, nargout=0)

    all_funcs = eng.genpath("D:/Poly_MSc_Code/libraries_and_toolboxes/sepia")
    eng.addpath(all_funcs, nargout=0)

    path_to_MEDI_tb = "D:/Poly_MSc_Code/libraries_and_toolboxes/toolboxes/MEDI_toolbox"
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
    global gm_mask_data, wm_mask_data, iter_folder
    gm_mask_img = nib.load(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing/gm_mask_crop.nii.gz")
    gm_mask_data = gm_mask_img.get_fdata()

    wm_mask_img = nib.load(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing/wm_mask_crop.nii.gz")
    wm_mask_data = wm_mask_img.get_fdata()

    iter_folder = rf"e:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params\BGFR_tests\iter_VSHARP/{test_fn}"
   
    if os.path.exists(iter_folder) and len(os.listdir(iter_folder)) > 0:
        print("Folder already exists and is not empty. Please delete the folder or choose a different name.")
        sys.exit(1)
  
    print("GM and WM masks loaded successfully.")

def load_groun_truth_data():
    global wb_gt_csf_ref_swiss_crop_fm_Hz_data
    wb_gt_csf_ref_swiss_crop_fm_Hz_data = nib.load(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\local_field_groud_truth\ref_csf/wb_gt_csf_ref_swiss_crop_fm_Hz.nii.gz").get_fdata()
    # This loads the Ground truth image with the Swiss Acq. Parameters FOV
    print("Ground truth local field loaded")

def vsharp_optimizer(x):
    global counter

    #matrix_Size = [301, 351, 128]
    #voxelSize = [0.976562, 0.976562, 2.344]

    max_radii = x.get_coord(0)
    min_radii = x.get_coord(1)
    
    iteration_fn = f"lbv_run{counter}/"

    output_fn = str(os.path.join(iter_folder,iteration_fn+"Sepia"))

    if not os.path.exists(output_fn):
        os.makedirs(output_fn)
        print("Created folder for new iteration #",counter)
    
    print("Output FN used:", output_fn)

    custom_fm_path = str(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params\fm_tests\test2_apply_msk/B0.nii")
    custom_header_path = str(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params/custom_qsm_sim.mat")
    mask_filename = str(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params/cord_mask_crop.nii.gz")
    
    in1 = custom_fm_path
    in2 = ""
    in3 = ""
    in4 = custom_header_path

    create_local_field(in1, in2, in3, in4, output_fn, mask_filename, max_radii, min_radii)
    # Import local field for RMSE calculation
    new_local_field_path = os.path.join(iter_folder,iteration_fn + "Sepia_localfield.nii.gz")
    
    print("Local field import from:", new_local_field_path)

    local_field_img = nib.load(new_local_field_path)
    local_field_data = local_field_img.get_fdata()

    # Now, we compute the difference between current local field with the Ground Truth
    pixel_wise_difference = wb_gt_csf_ref_swiss_crop_fm_Hz_data - local_field_data
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

    print(f"Iter {counter}: Max radii={max_radii}, Min radii ={min_radii}, GM-WM RMSE={objective_value}")


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
    "BB_OUTPUT_TYPE OBJ",
    "MAX_BB_EVAL 100",
    "DISPLAY_DEGREE 2",
    "DISPLAY_ALL_EVAL false",
    "DISPLAY_STATS BBE OBJ"
]
# For VSHARP the x0 should be [max_radii, min_radii]
x0 = [10, 3] # Recommended by SEPIA (for brain)

lb = [0, 1]

ub=[40, 39]

counter = 0

configure_experiment_run("RMSE_test3_100_evals_w_jsonsidecar")
load_groun_truth_data()

result = nomad.optimize(vsharp_optimizer, x0, lb, ub, nomad_params)

fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
output = "\n".join(fmt)
print("\nNOMAD results \n" + output + " \n")
