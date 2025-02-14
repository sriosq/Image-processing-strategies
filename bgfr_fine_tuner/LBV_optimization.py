#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
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
tolerance = 0.0001
depth = 5
peel = 2

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

matrixSize = [101,171,141]
voxelSize = [0.9766, 0.9766, 2.3440]

output_basename =  str("E:\msc_data\sc_qsm\data\cropped\piece-wise\simulation\TE_1_weird_40/bgfr_sepia/something/Sepia")
mask_filename = str("E:\msc_data\sc_qsm\data\cropped\piece-wise\simulation\canal_crop2.nii.gz")

in1 = "E:\msc_data\sc_qsm\data\cropped\piece-wise\simulation\TE_1_weird_40/romeo_tests/test3_right_masked\B0.nii"
in2 = ""
in3 = ""
in4 = "E:/msc_data/sc_qsm/data/cropped/piece-wise/simulation/TE_1_weird_40/header_qsm_tsting_hcrop2.mat"


eng.python_wrapper(in1, in2, in3, in4 , output_basename, mask_filename, bfr_params, nargout = 0)