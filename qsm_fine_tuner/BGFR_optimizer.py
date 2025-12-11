#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
import nibabel as nib
import numpy as np
import sys
import json

# Here I join the two functions to create a single function that can be used for both LBV, PDF, VHSARP and the other methods available

# This 

def create_local_field(in1, in2, in3, in4 , output_basename, mask_filename, method, tol=None, depth=None, peel=None, max_radii=None, min_radii=None, iteration=None, padSize=None):
    """ 
    Create a local field using SEPIA and the specified method.
    Parameters: 

    """
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
    if method == "LBV":
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

    # VSHARP parameters
    if method == "VSHARP":
            #
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
        
    # PDF parameters
    if method == "PDF":
        bfr_params = {  # Example method name
        
        'general' : {
            'isBET' : '0',
            'isInvert':'0',
            'isRefineBrainMask' : '0'
        },
        'bfr':{
        'method': "PDF",
        "refine_method" : "None",
        "refine_order" : 4,
        'erode_radius': 0,
      'erode_before_radius': 0,
        'tol': tolerance,
        'iteration': iteration,
        'padSize': padSize}
        }
    # Call the SEPIA function
    eng.python_wrapper(in1, in2, in3, in4 , method, output_basename, mask_filename, bfr_params, nargout = 0)
    print("Local Field Created! Calculate metrics and update parameters!")

def configure_experiment_run(test_fn, gm_mask_path, wm_mask_path, iter_folder_path):
    global gm_mask_data, wm_mask_data, iter_folder
    gm_mask_img =  nib.load(gm_mask_path)
    gm_mask_data = gm_mask_img.get_fdata()

    wm_mask_img =  nib.load(wm_mask_path)
    wm_mask_data = wm_mask_img.get_fdata()

    iter_folder =os.path.join(iter_folder_path, test_fn)
    if os.path.exists(iter_folder) and len(os.listdir(iter_folder)) > 0:
        print("Folder already exists and is not empty. Please delete the folder or choose a different name.")
        sys.exit(1)

    print("GM and WM masks loaded successfully.")

def load_groun_truth_data(path_to_gt_local_field):
    global wb_gt_csf_ref_swiss_crop_fm_Hz_data
    wb_gt_csf_ref_swiss_crop_fm_Hz_data = nib.load(path_to_gt_local_field).get_fdata()
    # This loads the Ground truth image with the Swiss Acq. Parameters FOV
    print("Ground truth local field loaded")

def load_sepia_inputs(custom_fm_path, custom_header_path, mask_filename_path):
    global in1, in2, in3, in4, mask_path
    in1 = custom_fm_path
    in4 = custom_header_path
    in2 = ""
    in3 = ""
    mask_path = mask_filename_path



def lbv_optimizer(x):
    global counter

    tolerance = x.get_coord(0)
    depth = x.get_coord(1)
    peel = x.get_coord(2)
    
    iteration_fn = f"lbv_run{counter}/"

    output_fn = str(os.path.join(iter_folder,iteration_fn+"Sepia"))

    if not os.path.exists(output_fn):
        os.makedirs(output_fn)
        print("Created folder for new iteration #",counter)
    
    print("Output FN used:", output_fn)

    create_local_field(in1, in2, in3, in4 , output_fn, mask_path, "LBV", tol=tolerance, depth=depth, peel=peel)
