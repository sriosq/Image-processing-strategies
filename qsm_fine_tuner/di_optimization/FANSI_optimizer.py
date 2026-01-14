#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
import nibabel as nib
import numpy as np
import sys
import json
import time  

def create_chimap(in1, in2, in3, in4 , output_basename, mask_filename, tol, maxiter, lmbda, mu1, mu2, solver, constraint, gmode, isWeakHarmonic, beta, muh):
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

    #  For FANSI there are some parameters that we can set and test per optimization:
    # One is the Solver: Non-linear or linear
    # Second is constraint: TV or TGV
    # Third is Gradient mode: Vector field, L1, L2 or None
    # If it is Weak harmonic we have more parameters to set:
    # Harmonic constraint and harmonic consistency
    # If Weak Harmonic is enabled, we cannot have None as the gradient mode!


    dipole_inv_params = {  
    
    'general' : {
        'isBET' : '0',
        'isInvert':'0',
        'isRefineBrainMask' : '0'

    },
    'qsm':{
    'reference_tissue': "Brain mask",
    "method": "FANSI",
    'tol': tol,
    'maxiter': maxiter,
    'lambda': lmbda, # Gradient penality
    'mu1': mu1, # Gradient consisntency
    'mu2': mu2, # Fidelity consistency
    'solver': solver,
    'constraint': constraint,   
    'gradient_mode': gmode,
    'isGPU': 0,
    'isWeakHarmonic': isWeakHarmonic,
    'beta': beta, # Harmonic constraint
    'muh': muh # Harmonic consistency
    }

}

    eng.python_wrapper(in1, in2, in3, in4 , 'FANSI', output_basename, mask_filename, dipole_inv_params, nargout = 0)
    print("Chi map! Calculate metrics and update parameters!")


def configure_experiment_run(test_fn, first_line="Optimization results: "):
    global gm_mask_data, wm_mask_data, iter_folder, txt_file_path
    gm_mask_img = nib.load(r"E:\msc_data\sc_qsm\final_gauss_sims/masks/sc_gm_crop.nii.gz")
    gm_mask_data = gm_mask_img.get_fdata()

    wm_mask_img = nib.load(r"E:\msc_data\sc_qsm\final_gauss_sims/masks/sc_wm_crop.nii.gz")
    wm_mask_data = wm_mask_img.get_fdata()

    print("GM and WM masks loaded successfully.")
    iter_folder = rf"E:\msc_data\sc_qsm\final_gauss_sims\November_2025\chi_mapping_opt/snr_60/iter_FANSI/weakH_off/{test_fn}"
   
    if os.path.exists(iter_folder) and len(os.listdir(iter_folder)) > 0:
        print("Folder already exists and is not empty. Please delete the folder or choose a different name.")
        sys.exit(1)
    else:
        os.makedirs(iter_folder, exist_ok=True)
        print("Experiment folder created!")

    txt_file_path = rf"E:\msc_data\sc_qsm\final_gauss_sims\November_2025\chi_mapping_opt/snr_60/iter_FANSI/weakH_off/{test_fn}.txt"
    with open(txt_file_path, 'w') as file:
        first_line_txt =  first_line + "\n"
        file.write(first_line_txt)

def load_groun_truth_chidist_data():
    global chimap_ref_sc_avg_
    # Lets first load the susceptibility ground truth map:
    #ground_truth_abs_chimap_data = nib.load(r"E:\msc_data\sc_qsm\final_gauss_sims\July_2025\ground_truth_data\bgfr_gt_ref_avg_sc_gauss_chi_dist_crop.nii.gz").get_fdata()
    # Now we need to use the average of the spinal cord mask because this is what SEPIA averages to with the mask
    #sc_mask_data = nib.load(r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus/cord_mask_crop.nii.gz").get_fdata()

    #avg_chi_sc_val = np.mean(ground_truth_abs_chimap_data[sc_mask_data==1])
    #print("Average chi value in spinal cord with std: ", avg_chi_sc_val)

    # Now apply the offset to the ground truth map
    #chimap_ref_sc_avg_ = ground_truth_abs_chimap_data - avg_chi_sc_val
    # Or load the already referenced map

    chimap_ref_sc_avg_ = nib.load(r"E:\msc_data\sc_qsm\final_gauss_sims\November_2025\ground_truth_data\di_gt_ref_avg_sc_gauss_chi_dist_crop.nii.gz").get_fdata()

    print("Ground truth susceptibility map loaded")

def log_best_solution(obj_value, iteration, tol, maxiter, lmbda, mu1, mu2, solver, constraint, gmode, isWeakH, beta, muh, gm_rmse, wm_rmse):
    global best_obj_value
    total_rmse = gm_rmse + wm_rmse
    if obj_value <= best_obj_value:
        if obj_value == best_obj_value:
            print("Found a solution with the same objective value, but different parameters.")
            with open(txt_file_path, 'a') as file:
                file.write(f"Iterration: {iteration}: OBJ {obj_value} // Tolerance: {tol}, #of Iter: {maxiter}, Lamba:{lmbda}, // Gradient Consistency: {mu1}, Fidelity consistency: {mu2}, Solver: {solver}, Constraint: {constraint}, Gradient mode: {gmode}, WeakH: {isWeakH}, Harmonic constraint: {beta}, Harmonic consistency: {muh} // GM RMSE: {gm_rmse}, WM RMSE: {wm_rmse} | RMSE: {total_rmse} \n")
                           

        best_obj_value = obj_value
        print(f"New best solution found: {obj_value}")
        
        with open(txt_file_path, 'a') as file:
                file.write(f"Iterration: {iteration}: OBJ {obj_value} // Tolerance: {tol}, #of Iter: {maxiter}, Lamba:{lmbda}, // Gradient Consistency: {mu1}, Fidelity consistency: {mu2}, Solver: {solver}, Constraint: {constraint}, Gradient mode: {gmode}, WeakH: {isWeakH}, Harmonic constraint: {beta}, Harmonic consistency: {muh} // GM RMSE: {gm_rmse}, WM RMSE: {wm_rmse} | RMSE: {total_rmse} \n")


def FANSI_optimizer_weakH_off(x):
    global counter

    #matrix_Size = [301, 351, 128]
    #voxelSize = [0.976562, 0.976562, 2.344]
    # 
    # In the FANSI code, we find some relationships between parameters:
    # Magnitude recommended for non-linear solver (we are using non-linear, so magnitude is preferred)
    # alpha1 2e-4
    # alpha0 = 2*alpha1 recommended

    tol = 0.1 # Tolerance default is 0.1, heuristic definition 0.05
    maxiter = 150 # Default is 150, extended to 300 for better convergence
    lmbda = x.get_coord(0) # lambda is Gradient L1 penalty, this is how SEPIA handles it, I will submit a change to their repo cos its confusing
    mu1 = x.get_coord(1) # Gradient consistency
    mu2 = 1 # Fidelity consistency and should remain 1 - this just uses the magnitude
    solver = 'Non-linear'  # 'Non-linear' or 'Linear', we fix to non-linear
    constraint = 'TV'  # 'TGV' or 'TV', we fix to TV
    gmode = 'L1'  # 'Vector field', 'L1', 'L2',  or 'None' 
    # Test and assess diffenrent gradient modes both in-vivo and in-sillico

    # First we do with weak harmonic off, phantom shouldn't need weak ON?
    # Perhaps after adding noise then we could use it, I would try a couple optimizers with ON and assess if there is any benefit for the phantom
    # Then we add x.get_coord for both beta and muh
    isWeakHarmonic = '0'  # Fixed and unused when isWeakHarmonic = 0
    beta = 150   # Harmonic constraint
    muh = 3  # Harmonic consistency
    
    iteration_fn = f"FANSI_run{counter}/"

    output_fn = str(os.path.join(iter_folder,iteration_fn+"Sepia"))

    if not os.path.exists(output_fn):
        os.makedirs(output_fn)
        print("Created folder for new iteration #",counter)
    
    print("Output FN used:", output_fn)

    gt_local_field_path =str(r"E:\msc_data\sc_qsm\final_gauss_sims\November_2025\ground_truth_data\noisy\cropped\di_in_gt_ref_avg_sc_lf_Hz_snr60_crop.nii.gz") 
    # Instead of using the output of the best optimized local field, we want to optimize the algorithm with the best possible local field
    # This is the gt susceptibility map convoluted with the dipole kernel that gives us the GT LF for the BGFR optimization!
    custom_header_path = str(r"E:\msc_data\sc_qsm\final_gauss_sims\November_2025\mrsim_outputs\qsm_sc_phantom_custom_params.mat")
    mask_filename = str(r"E:\msc_data\sc_qsm\final_gauss_sims\masks\only_sc_crop.nii.gz")# str(r"E:\msc_data\sc_qsm\final_gauss_sims/masks\qsm_processing_msk_crop.nii.gz")

    # Some algorithms use the magnitude for weighting! Should be input #2
    gauss_sim_ideal_mag_path = str(r"E:\msc_data\sc_qsm\final_gauss_sims\November_2025\mrsim_outputs\custom_params_snr_74\gauss_crop_sim_mag_pro.nii.gz")
    # Some algorithms need weigths for noise distribution, we can use the mask as a replacement if we want fair comparison with other algorithms that dont use it
    sepia_weights_path = mask_filename
    
    in1 = gt_local_field_path
    in2 = gauss_sim_ideal_mag_path 
    in3 = sepia_weights_path
    in4 = custom_header_path

    create_chimap(in1, in2, in3, in4, output_fn, mask_filename, tol, maxiter, lmbda, mu1, mu2, solver, constraint, gmode, isWeakHarmonic, beta, muh)
    # Import local field for RMSE calculation
    new_chimap_path = os.path.join(iter_folder, iteration_fn + "Sepia_chimap.nii.gz")
    
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

    # Objective: Maximize the difference between GM and WM means
    # PyNomad minimizes, so return negative to maximize

    objective_value = gm_rmse + wm_rmse

    log_best_solution(objective_value, counter, tol, maxiter, lmbda, mu1, mu2, solver, constraint, gmode, isWeakHarmonic, beta, muh, gm_rmse, wm_rmse)

    print(f"Iter {counter}: GM+WM RMSE = {objective_value}")


    # Data to save

    sidecar_data = {
        "iteration": counter,
        "tolerance": tol,
        "max_iterations": maxiter,
        "Gradient penalty": lmbda, 
        "Gradient consistency": mu1,
        "Fidelity consistency ": mu2,
        "solver": solver,
        "constraint": constraint,
        "gradient_mode": gmode,
        "isWeakHarmonic": isWeakHarmonic,
        "Harmonic constraint": beta,
        "Harmonic consistency": muh,
        "gm_rmse": gm_rmse,
        "wm_rmse": wm_rmse,
        "objective_value": objective_value
    }

    # We want this to be saved in the precise run so:
    json_filename = os.path.join(iter_folder, iteration_fn, "sidecar_data.json")
    with open(json_filename, 'w') as json_file:
        json.dump(sidecar_data, json_file, indent=4)
    print("Sidecar data saved to:", json_filename)

    # Increase counter
    counter += 1

    rawBBO = str(objective_value)
    x.setBBO(rawBBO.encode("UTF-8"))

    return 1

#############################################################################################################################################

nomad_params_weak_OFF = [
    "DIMENSION 2",
    "BB_INPUT_TYPE (R R)", # lmbda, mu1, mu2, beta and muh
    "BB_OUTPUT_TYPE OBJ",
    "MAX_BB_EVAL 10",
    "DISPLAY_DEGREE 2",
    "DISPLAY_ALL_EVAL false",
    "DISPLAY_STATS BBE OBJ"
]

nomad_params_weak_ON = [
    "DIMENSION 5",
    "BB_INPUT_TYPE (R R R R R)", # lmbda, mu1, mu2, beta and muh
    "BB_OUTPUT_TYPE OBJ",
    "MAX_BB_EVAL 20",
    "DISPLAY_DEGREE 2",
    "DISPLAY_ALL_EVAL false",
    "DISPLAY_STATS BBE OBJ"
]
# After careful revition of the paper + the code and the theory for FANSI, we'll fix the solver as well as the constraint
# The gradient mode can also be fixed but studying how different modes affect the outcome would be interesting
    # L1 promotes sparsity, L2 smoother gtradients, Vector field is the most complete but also computationally heavy (?) 
# For the optimization we want realistic final values so boundaries are selected thinking on translation of in-sillico to in-vivo scenarios
# The fidelity terms, ergo, mu2 - fidelity consistency that starts at 1 could go from 0.01 to 10, if to high noisy reconstruction, to low may not converge 
# then, weight is taken from the magnitude, because our simulations are realistic I don't see why not giving it to the solver, the tolerance is fixed

# The, regularization terms: alpha - gradient l1 penalty, mu1 gradient consistency

# 

# Begin:
start_time = time.time()
# Parameters are: [lmbda, mu1, mu2, beta, muh] beta and muh are only used if isWeakHarmonic = 1
x0_weakOFF = [0.0002, 0.02] # Recommended by SEPIA (for brain)

lb_weakOFF = [0.0000001, 0.0000002]

ub_weakOFF = [0.1, 20]

######################################################################################
x0_weakON = [0.0002, 0.02, 1, 150, 3] # Recommended by SEPIA (for brain)

lb_weakON = [0.000001, 0.1, 0.001, 1, 0.1]

ub_weakON = [0.01, 2, 10, 300, 30]

counter = 0
# In total there will be 24 runs:

# I_non-linear_TGV_vector_field_weakH_off Done
# II_linear_TGV_vector_field_weakH_off Done
# III_non-linear_TGV_L1_weakH_off
# IV_linear_TGV_L1_weakH_off
# V_non-linear_TGV_L2_weakH_off
# VI_linear_TGV_L2_weakH_off

# VII_non-linear_TV_L1_weakH_off
# VIII_linear_TV_L1_weakH_off
# IX_non-linear_TV_L2_weakH_off
# X_linear_TV_L2_weakH_off
# XI_non-linear_TV_vector_field_weakH_off
# XII_linear_TV_vector_field_weakH_off

# Then repeat all again with weak harmonic on

# XIII_non-linear_TGV_vector_field_weakH_on
# XIV_linear_TGV_vector_field_weakH_on
# XV_non-linear_TGV_L1_weakH_on
# XVI_linear_TGV_L1_weakH_on
# XVII_non-linear_TGV_L2_weakH_on
# XVIII_linear_TGV_L2_weakH_on
# XIX_non-linear_TV_L1_weakH_on
# XX_linear_TV_L1_weakH_on
# XXI_non-linear_TV_L2_weakH_on
# XXII_linear_TV_L2_weakH_on
# XXIII_non-linear_TV_vector_field_weakH_on
# XXIV_linear_TV_vector_field_weakH_on

# After careful consideration and testing, we decide to use only Non-Linear solver
# And begin with using TV 

first_line = "Optimization results for FANSI @60 SNR, Non-Linear TV L1, weak harmonics OFF:"
configure_experiment_run("heuristic_nL_TV_L1_weakH_off/test2_for_def_rmse", first_line)
best_obj_value = float('inf')
load_groun_truth_chidist_data()

result = nomad.optimize(FANSI_optimizer_weakH_off, x0_weakOFF, lb_weakOFF, ub_weakOFF, nomad_params_weak_OFF)


fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
output = "\n".join(fmt)
print("\nNOMAD results \n" + output + " \n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Optimization complete in: {elapsed_time:.3f} seconds")
