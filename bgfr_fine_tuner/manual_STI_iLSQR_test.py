#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
import nibabel as nib

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


#  Parameters for STI version of iLSQR
thr = 0.01
max_iters = 100
tolerance_1 = 0.01
tolerance_2 = 0.001
pad_size = 12

dipole_inv_params = {  # Example method name
    
    'general' : {
        'isBET' : '0',
        'isInvert':'0',
        'isRefineBrainMask' : '0'
    },
    'qsm':{
    'reference_tissue': "Brain mask",
    "method": "TKD",
    'threshold':thr}

}

matrixSize = [101,171,141]
voxelSize = [0.9766, 0.9766, 2.3440]
counter = 100
iteration_fn = f"tkd_run_test_manual_after_reset/"
iter_folder = r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params\dipole_inversion_tests\manual_testing_iLSQR_after_reset"


output_basename = str(os.path.join(iter_folder,iteration_fn+"Sepia"))
if not os.path.exists(output_basename):
    os.makedirs(output_basename)
    print("New folder created")

print("Output FN used:", output_basename)

#output_basename =  str("E:\msc_data\sc_qsm\sim_data\cropped\piece-wise\simulation\TE_1_weird_40/bgfr_sepia/something/Sepia")
#mask_filename = str("E:\msc_data\sc_qsm\sim_data\cropped\piece-wise\simulation\canal_crop2.nii.gz")

#in1 = "E:\msc_data\sc_qsm\sim_data\cropped\piece-wise\simulation\TE_1_weird_40/romeo_tests/test3_right_masked\B0.nii"
best_local_fiel_path = str(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params\BGFR_tests\iter_PDF\RMSE_test2_100_evals\lbv_run94/Sepia_localfield.nii.gz")
custom_header_path = str(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params/custom_qsm_sim.mat")
mask_filename = str(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params/cord_mask_crop.nii.gz")
    
in1 = best_local_fiel_path

in4 = custom_header_path

in2 = ""
in3 = ""
#in4 = "E:/msc_data/sc_qsm/sim_data/cropped/piece-wise/simulation/TE_1_weird_40/header_qsm_tsting_hcrop2.mat"


eng.python_wrapper(in1, in2, in3, in4 , 'TKD', output_basename, mask_filename, dipole_inv_params, nargout = 0)

print("Test loading Chi Map")

new_local_chimap = os.path.join(iter_folder,iteration_fn + "Sepia_chimap.nii.gz")
chimap_img = nib.load(new_local_chimap)
chimap_data= chimap_img.get_fdata()

print("Loaded succesufully!")

