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


#  VSHARP parameter is max and min radius
max_radii = 10
min_radii = 3
radius_list = list(range(10,2,-1))
radius_matlab = matlab.double(radius_list)

bfr_params = {  # Example method name
    
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

matrixSize = [101,171,141]
voxelSize = [0.9766, 0.9766, 2.3440]
counter = 100
iteration_fn = f"lbv_run{counter}/"
iter_folder = r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params\BGFR_tests\manual_testing_vsharp"


output_basename = str(os.path.join(iter_folder,iteration_fn+"Sepia"))
if not os.path.exists(output_basename):
    os.makedirs(output_basename)
    print("New folder created")

print("Output FN used:", output_basename)

#output_basename =  str("E:\msc_data\sc_qsm\sim_data\cropped\piece-wise\simulation\TE_1_weird_40/bgfr_sepia/something/Sepia")
#mask_filename = str("E:\msc_data\sc_qsm\sim_data\cropped\piece-wise\simulation\canal_crop2.nii.gz")

#in1 = "E:\msc_data\sc_qsm\sim_data\cropped\piece-wise\simulation\TE_1_weird_40/romeo_tests/test3_right_masked\B0.nii"
custom_fm_path = str(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params\fm_tests\test2_apply_msk/B0.nii")
custom_header_path = str(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params/custom_qsm_sim.mat")
mask_filename = str(r"E:\msc_data\sc_qsm\Swiss_data\march_25_re_process\MR_simulations\sim_data\QSM_processing\mrsim_outputs\custom_acq_params/cord_mask_crop.nii.gz")
    
in1 = custom_fm_path

in4 = custom_header_path

in2 = ""
in3 = ""
#in4 = "E:/msc_data/sc_qsm/sim_data/cropped/piece-wise/simulation/TE_1_weird_40/header_qsm_tsting_hcrop2.mat"


eng.python_wrapper(in1, in2, in3, in4 , 'VSHARP', output_basename, mask_filename, bfr_params, nargout = 0)

print("Test loading Local Field")

new_local_field_path = os.path.join(iter_folder,iteration_fn + "Sepia_localfield.nii.gz")
local_field_img = nib.load(new_local_field_path)
local_field_data = local_field_img.get_fdata()

print("Loaded succesufully!")

