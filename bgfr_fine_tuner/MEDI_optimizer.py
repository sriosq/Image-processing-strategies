#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
import nibabel as nib
import numpy as np
import sys
import json
import time  

def create_chimap(in1, in2, in3, in4 , output_basename, mask_filename, lammbda, percentage, radius):
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

    #  For NDI: tolerance, max # of iterations and step size
    zeropad_size = [0,0,0]

    dipole_inv_params = {  
    
    'general' : {
        'isBET' : '0',
        'isInvert':'0',
        'isRefineBrainMask' : '0'
    },
    'qsm':{
    'reference_tissue': "Brain mask",
    "method": "MEDI",
    'lambda': 0,
    'wData': 1,
    'percentage': percentage,
    'zeropad': zeropad_size,
    'isSMV': 1,
    'radius': radius,
    'merit': 1,
    'isLambdaCSF': 0,
    'lambdaCSF': 100} # This is a boolean option, test 1 time with 0 then turn on in SEPIA and compare

}

    eng.python_wrapper(in1, in2, in3, in4 , 'MEDI', output_basename, mask_filename, dipole_inv_params, nargout = 0)
    print("Chi map! Calculate metrics and update parameters!")


def configure_experiment_run(test_fn):
    global gm_mask_data, wm_mask_data, iter_folder, txt_file_path
    gm_mask_img = nib.load(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus/gm_mask_crop.nii.gz")
    gm_mask_data = gm_mask_img.get_fdata()

    wm_mask_img = nib.load(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus/wm_mask_crop.nii.gz")
    wm_mask_data = wm_mask_img.get_fdata()

    print("GM and WM masks loaded successfully.")
    iter_folder = rf"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\dipole_inversion_tests\iter_MEDI/{test_fn}"
   
    if os.path.exists(iter_folder) and len(os.listdir(iter_folder)) > 0:
        print("Folder already exists and is not empty. Please delete the folder or choose a different name.")
        sys.exit(1)
    else:
        os.makedirs(iter_folder, exist_ok=True)
        print("Experiment folder created!")

    txt_file_path = rf"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\dipole_inversion_tests\iter_MEDI/{test_fn}.txt"
    with open(txt_file_path, 'w') as file:
        file.write("Optimization results.\n")

def load_groun_truth_chidist_data():
    global chimap_ref_sc_avg_
    # Lets first load the susceptibility ground truth map:
    ground_truth_abs_chimap_data = nib.load(r"E:\msc_data\sc_qsm\new_gauss_sims\sim_inputs\chi_to_fm_ppm/gauss_chi_sc_phantom_swiss_crop.nii.gz").get_fdata()
    # Now we need to use the average of the spinal cord mask because this is what SEPIA averages to with the mask
    sc_mask_data = nib.load(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus/cord_mask_crop.nii.gz").get_fdata()

    avg_chi_sc_val = np.mean(ground_truth_abs_chimap_data[sc_mask_data==1])
    print("Average chi value in spinal cord with std: ", avg_chi_sc_val)

    # Now apply the offset to the ground truth map
    chimap_ref_sc_avg_ = ground_truth_abs_chimap_data - avg_chi_sc_val
    print("Ground truth susceptibility map loaded")

def log_best_solution(obj_value, iteration, lmbda, percentage, radius, gm_rmse, wm_rmse):
    global best_obj_value
    total_rmse = gm_rmse + wm_rmse
    if obj_value <= best_obj_value:
        if obj_value == best_obj_value:
            print("Found a solution with the same objective value, but different parameters.")
            with open(txt_file_path, 'a') as file:
                file.write(f"Iteration: {iteration}: OBJ {obj_value} // Lambda: {lmbda}, Percentage: {percentage}, SMV radius: {radius}, GM RMSE: {gm_rmse}, WM RMSE: {wm_rmse} | RMSE: {total_rmse} \n")

        best_obj_value = obj_value
        print(f"New best solution found: {obj_value}")
        
        with open(txt_file_path, 'a') as file:
            file.write(f"Iteration: {iteration}: OBJ {obj_value} // Lambda: {lmbda}, Percentage: {percentage}, SMV radius: {radius}, GM RMSE: {gm_rmse}, WM RMSE: {wm_rmse} | RMSE: {total_rmse} \n")


def MEDI_optimizer(x):
    global counter

    #matrix_Size = [301, 351, 128]
    #voxelSize = [0.976562, 0.976562, 2.344]
    # lammbda, percentage, radius
    lmbda = x.get_coord(0)
    percentage = x.get_coord(1)
    smv_radius = x.get_coord(2)
    
    iteration_fn = f"MEDI_run{counter}/"

    output_fn = str(os.path.join(iter_folder,iteration_fn+"Sepia"))

    if not os.path.exists(output_fn):
        os.makedirs(output_fn)
        print("Created folder for new iteration #",counter)
    
    print("Output FN used:", output_fn)
    # Best local field is using SHARP!
    best_local_fiel_path = str(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\bgfr_opt\iter_SHARP\RMSE_test1_mskd_fm_200_evals\sharp_run72/Sepia_localfield.nii.gz")
    custom_header_path = str(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal/custom_qsm_sim.mat")
    mask_filename = str(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus/cord_mask_crop.nii.gz")

    # Some algorithms use the magnitude for weighting! Should be input #2
    gauss_sim_ideal_mag_path = str(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\gauss_crop_sim_mag_pro.nii.gz")
    
    in1 = best_local_fiel_path
    in2 = gauss_sim_ideal_mag_path 
    in3 = ""
    in4 = custom_header_path

    create_chimap(in1, in2, in3, in4, output_fn, mask_filename, tol, maxiter, stepSize, isGPU)
    # Import local field for RMSE calculation
    new_chimap_path = os.path.join(iter_folder,iteration_fn + "Sepia_chimap.nii.gz")
    
    print("Chimap imported from:", new_chimap_path)

    chimap_img = nib.load(new_chimap_path)
    chimap_data = chimap_img.get_fdata()

    # Now, we compute the difference between current local field with the Ground Truth
    pixel_wise_difference = chimap_ref_sc_avg_ - chimap_data
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
    log_best_solution(objective_value, counter, tol, maxiter, stepSize, gm_rmse, wm_rmse)

    print(f"Iter {counter}: Lambda = {lmbda}, Percentage = {percentage}, SMV radius = {radius} GM+WM RMSE = {objective_value}")

    # Data to save
    sidecar_data = {
        'iteration': counter,
        'Lambda' = float(lmbda),
        'Percentage': float(percentage),
        'SMV radius': float(radius),
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
    "BB_INPUT_TYPE (R R R)",
    "BB_OUTPUT_TYPE OBJ",
    "MAX_BB_EVAL 600",
    "DISPLAY_DEGREE 2",
    "DISPLAY_ALL_EVAL false",
    "DISPLAY_STATS BBE OBJ"
]
# For MEDI the only parameters to optimize are the lambda, percentage and radius
# The lambda is the regularization parameter, percentage is the percentage of the local field to use and radius is the SMV radius
# Begin:
start_time = time.time()
x0 = [1, 90, 5] # Recommended by SEPIA (for brain)

lb = [0.000001, 1, 1]

ub = ['+inf', 99, '+inf']

counter = 0

configure_experiment_run("RMSE_test1_with_mag_600_evals")
best_obj_value = float('inf')
load_groun_truth_chidist_data()

result = nomad.optimize(MEDI_optimizer, x0, lb, ub, nomad_params)


fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
output = "\n".join(fmt)
print("\nNOMAD results \n" + output + " \n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Optimization complete in: {elapsed_time:.3f} seconds")
