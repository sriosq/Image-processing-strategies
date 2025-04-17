#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
import nibabel as nib
import numpy as np
import sys

def create_local_field(in1, in2, in3, in4 , output_basename, mask_filename, tol, depth, peel):
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

    #  LBV Parameters
    tolerance = tol
    depth = depth
    peel = peel

    bfr_params = {  # Example method name
    
    'general' : {
        'isBET' : '0',
        'isInvert':'0',
        'isRefineBrainMask' : '0'
    },
    'bfr':{
    'method': "LBV",
    'tol': float(tolerance),
    'depth': float(depth),
    'peel': float(peel),
    "refine_method" : "None",
    "refine_order" : 4,
    'erode_radius': 0,
  'erode_before_radius': 0}

    }   

    eng.python_wrapper(in1, in2, in3, in4 , 'LBV', output_basename, mask_filename, bfr_params, nargout = 0)
    print("Local Field Created! Calculate metrics and update parameters!")


def load_masks(test_fn):
    global gm_mask_data, wm_mask_data, iter_folder
    gm_mask_img = nib.load(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing/gm_mask_crop.nii.gz")
    gm_mask_data = gm_mask_img.get_fdata()

    wm_mask_img = nib.load(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing/wm_mask_crop.nii.gz")
    wm_mask_data = wm_mask_img.get_fdata()

    iter_folder = rf"e:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params\BGFR_tests\iter_LBV/{test_fn}"
    
    print("GM and WM masks loaded successfully.")

def load_groun_truth_data():
    global wb_gt_csf_ref_swiss_crop_fm_Hz_data
    wb_gt_csf_ref_swiss_crop_fm_Hz_data = nib.load(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\local_field_groud_truth\ref_csf/wb_gt_csf_ref_swiss_crop_fm_Hz.nii.gz").get_fdata()
    # This loads the Ground truth image with the Swiss Acq. Parameters FOV
    print("Ground truth local field loaded")

def lbv_optimizer(x):
    global counter

    #matrix_Size = [301, 351, 128]
    #voxelSize = [0.976562, 0.976562, 2.344]

    tolerance = x.get_coord(0)
    depth = x.get_coord(1)
    peel = x.get_coord(2)
    
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

    create_local_field(in1, in2, in3, in4, output_fn, mask_filename, tolerance, depth, peel)
    
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

    print(f"Iter {counter}: Tolerance={tolerance}, Depth={depth}, Peel={peel}, GM-WM RMSE={objective_value}")

    rawBBO = str(objective_value)
    x.setBBO(rawBBO.encode("UTF-8"))

    return 1

#############################################################################################################################################

nomad_params = [
    "DIMENSION 3",
    "BB_OUTPUT_TYPE OBJ",
    "MAX_BB_EVAL 30",
    "DISPLAY_DEGREE 2",
    "DISPLAY_ALL_EVAL false",
    "DISPLAY_STATS BBE OBJ"
]

x0 = [0.0001,5,2]

lb = [0.000001, -1, 0.1]

ub=[1,5,5]

counter = 0

load_masks("RMSE_test1_30_evals")
load_groun_truth_data()

result = nomad.optimize(lbv_optimizer,x0,lb,ub,nomad_params)

fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
output = "\n".join(fmt)
print("\nNOMAD results \n" + output + " \n")
