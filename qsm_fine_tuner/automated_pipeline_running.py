import os
import nibabel as nib
import numpy as np 
import pandas as pd
import matlab.engine
import sys
import json
import time
import subprocess
import shlex

# In this function code we will run a pipeline of fieldmap -> chi map using our python SEPIA wrapper
# Based on a pipeline identifier using optimized or default pipelines

# Dictionary of pipelines:

# 1st pipeline, created towards end of october 2025 -> for ISMRM abstract uses PDF for BGFR and TKD for DI

def dictionary_of_pipelines():
    pipelines = {
        "mk1": ["opt_PDF", "opt_TKD"],
        "mk1_zero": ["def_PDF", "def_TKD"],
        'comp_bgfr':["def_pdf", "def_lbv", "opt_sharp", "opt_resharp", "opt_pdf", "opt_lbv"],
        'comp_di': ["def_tkd", "opt_tkd", "def_iLSQR",  "opt_iLSQR", "auto_iLSQR", 
                    "def_closedForm",  "opt_closedForm", "auto_closedForm", "def_fansi", "opt_fansi"] # For now MEDI manuall run
    }
    return pipelines

def signal_to_fm_Hz(mag_path, ph_path, TEs, outfn, msk_path=None, romeo_path="c:/romeo/romeo.jl", phs_offset_correction = None):

    TE_str = "[" + ", ".join(str(TE) for TE in TEs) + "]"

    # Build command as a LIST, not a string
    cmd = ["julia", romeo_path, "-p", ph_path, "-m", mag_path]

    if msk_path is not None:
        cmd += ["-k", msk_path, "-B", "-Q", "-t", TE_str, "-u", "-o", outfn]
    else:
        cmd += ["-B", "-Q", "-t", TE_str, "-o", outfn]

    if phs_offset_correction:
        print("Offset correction found, using: ", phs_offset_correction)
        cmd += ["--phase-offset-correction", phs_offset_correction]

    print("Running ROMEO with command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print(f"ROMEO completed. Output saved in: {outfn}")

def dict_of_algo_params(algo, step):
    bgfr_params = {
        'general' : {
            'isBET' : '0',
            'isInvert':'0',
            'isRefineBrainMask' : '0'
        },
        'bfr':{}
    }

    di_params = {
        'general' : {
            'isBET' : '0',
            'isInvert':'0',
            'isRefineBrainMask' : '0'
        },
        'qsm':{}
    }

    di_params = {  
    
        'general' : {
            'isBET' : '0',
            'isInvert':'0',
            'isRefineBrainMask' : '0'
        },
        'qsm':{}
    }

    ############## FOR BGFR automated comp running ##############

    if algo == "def_pdf":
        bgfr_params['bfr'] ={
            'method': "PDF",
            'tol': matlab.double(0.1),
            'iteration': matlab.double(50),
            'padSize': matlab.double(40),
            "refine_method" : "None",
            "refine_order" : 4,
            'erode_radius': 0,
            'erode_before_radius': 0
            }
        
    elif algo == "def_lbv":
        bgfr_params['bfr'] = {
            'method': "LBV",
            'tol': matlab.double(0.0001),
            'depth': matlab.double(5),
            'peel': matlab.double(2),
            "refine_order" : 4,
            'erode_radius': 0,
            'erode_before_radius': 0
            }
        
    elif algo == "opt_resharp":
        bgfr_params['bfr'] = {
            'method': "RESHARP",
            'radius': matlab.double(1),
            'depth': matlab.double(0.02),
            "refine_order" : 4,
            'erode_radius': 0,
            'erode_before_radius': 0
        }

    elif algo == "opt_pdf":
        bgfr_params['bfr'] = {
            'method': "PDF",
            'tol': matlab.double(0.001),
            'iteration': matlab.double(200),
            'padSize': matlab.double(34),
            "refine_method" : "None",
            "refine_order" : 4,
            'erode_radius': 0,
            'erode_before_radius': 0
        }

    elif algo == "opt_lbv":
        bgfr_params['bfr'] = {
            'method': "LBV",
            'tol': matlab.double(0.001),
            'depth': matlab.double(6),
            'peel': matlab.double(1),
            "refine_order" : 4,
            'erode_radius': 0,
            'erode_before_radius': 0
        
        }

    elif algo == "opt_sharp":
        bgfr_params['bfr'] = {
            'method': "SHARP",
            'radius': matlab.double(1),
            'threshold': matlab.double(0.03),
            "refine_order" : 4,
            'erode_radius': 0,
            'erode_before_radius': 0
        }
    
    ############## FOR DI automated comp running ##############
    elif algo == "def_tkd":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "TKD",
            'threshold': matlab.double(0.15)
        }

    elif algo == "opt_tkd":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "TKD",
            'threshold': matlab.double(0.024)
        }
                
    elif algo == "def_iLSQR":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "iLSQR",
            'tol': matlab.double(0.001),
            'maxiter': matlab.double(100),
            'lambda': matlab.double(0.13),
            'optimise': matlab.double(0)
        }
            
    elif algo == "opt_iLSQR":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "iLSQR",
            'tol': matlab.double(0.000001),
            'maxiter': matlab.double(57),
            'lambda': matlab.double(0.00001),
            'optimise': matlab.double(0)
        }

    elif algo == "auto_iLSQR":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "iLSQR",
            'tol': matlab.double(0.000001),
            'maxiter': matlab.double(57),
            'lambda': matlab.double(0.13),
            'optimise': matlab.double(1) # This should override the lambda value above
        }
            
    elif algo == "def_closedForm":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "Closed-form solution",
            'lambda': matlab.double(0.13),
            'optimise': matlab.double(0)

        }
            
    elif algo == "opt_closedForm":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "Closed-form solution",
            'lambda': matlab.double(0.010962),
            'optimise': matlab.double(0)

        }

    elif algo == "auto_closedForm":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "Closed-form solution",
            'lambda': matlab.double(0.13),
            'optimise': matlab.double(1) # This should override the lambda value above
        }
            
    elif algo == "def_fansi":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "FANSI",
            'tol': matlab.double(0.1),
            'maxiter': matlab.double(150),
            'lambda': matlab.double(0.0002),
            'mu1': matlab.double(0.02),
            'mu2': matlab.double(1),
            'solver': "Non-linear",
            'constraint': "TGV",
            "gradient_mode": "L2 norm",
            "isGPU": 0,
            "isWeakHarmonic": 0
        }
        
    elif algo == "opt_fansi":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "FANSI",
            'tol': matlab.double(0.05),
            'maxiter': matlab.double(300),
            'lambda': matlab.double(0.01173),
            'mu1': matlab.double(0.4542),
            'mu2': matlab.double(1),
            'solver': "Non-linear",
            'constraint': "TGV",
            "gradient_mode": "L2 norm",
            "isGPU": 0,
            "isWeakHarmonic": 0
        }
            
    elif algo == "def_medi":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "MEDI",
            'wData': matlab.double(1),
            'lambda': matlab.double(1000),
            'percentage': 90,
            'zeropad':  np.array([0,0,0]),
            'isSMV': 1,
            'radius': matlab.double(1),
            "merit": 0,
            "isLambdaCSF": 0,
            "lambdaCSF":100
        }
            
    elif algo == "opt_medi":
        di_params['qsm'] = {
            'reference_tissue': "Brain mask",
            "method": "MEDI",
            'wData': matlab.double(1),
            'lambda': matlab.double(10000),
            'percentage': 92,
            'zeropad':  np.array([0,0,0]),
            'isSMV': 0,
            'radius': matlab.double(1), # Doens't matter because we are removing it!
            "merit": 0,
            "isLambdaCSF": 0,
            "lambdaCSF":100
        }

    else:
        print(f"Couldn't find {algo} in algos, please check function!")

    if step == "bgfr":
        return bgfr_params
    if step == "di":
         return di_params


def get_algo_params(pipeline_id):
    if pipeline_id == "mk1":
        # This uses optimized PDF and TKD:
        bgfr_params = {
        'general' : {
            'isBET' : '0',
            'isInvert':'0',
            'isRefineBrainMask' : '0'
        },
        'bfr':{
            'method': "PDF",
            'tol': matlab.double(0.001),
            'iteration': matlab.double(200),
            'padSize': matlab.double(34),
            "refine_method" : "None",
            "refine_order" : 4,
            'erode_radius': 0,
            'erode_before_radius': 0}
        }

        
        dipole_inv_params = {  
    
        'general' : {
            'isBET' : '0',
            'isInvert':'0',
            'isRefineBrainMask' : '0'
        },
        'qsm':{
            'reference_tissue': "Brain mask",
            "method": "TKD",
            'threshold': matlab.double(0.0017)}
        }
        
    return bgfr_params, dipole_inv_params


def fieldmap_to_chimap_wrapper(fieldmap_path, header_path, mask_path, pipeline_id, outpath):
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
    
    pipelines = dictionary_of_pipelines()

    if pipeline_id  not in pipelines:
        raise ValueError(f"Pipeline {pipeline_id} not found on dictionary: /n {pipelines}")

    steps = pipelines[pipeline_id]
    outpath_local_field = os.path.join(outpath, pipeline_id, "localfield/Sepia")
    outpath_chi_map = os.path.join(outpath, pipeline_id, "chimap/Sepia")
    os.makedirs(outpath_local_field, exist_ok=True)
    os.makedirs(outpath_chi_map, exist_ok=True)

    bgfr_params_auto, di_params_auto = get_algo_params(pipeline_id)

    eng.python_wrapper(fieldmap_path, "", "", header_path , 'PDF', outpath_local_field, mask_path, bgfr_params_auto, nargout = 0)
    print("Local Field Created! Calculate metrics and update parameters!")

    # Now get the path to the local field to use as input for the dipole inversion step
    local_field_map_path =os.path.join(outpath, pipeline_id, "localfield", "Sepia_localfield.nii.gz")

    eng.python_wrapper(local_field_map_path, "", "", header_path , 'TKD', outpath_chi_map, mask_path, di_params_auto, nargout = 0)
    print("Chi map Created! Calculate metrics and update parameters!")

def bgfr_comp(fieldmap_path, header_path, mask_path, pipeline_id, outpath):

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
    
    pipelines = dictionary_of_pipelines()

    if pipeline_id  not in pipelines:
        raise ValueError(f"Pipeline {pipeline_id} not found on dictionary: /n {pipelines}")

    for algo in pipelines[pipeline_id]:
        print("Beginning BGFR algo comp automated running")
        print("Input algo: ", algo)
        bgfr_params = dict_of_algo_params(algo, "bgfr")
        algo_name =  bgfr_params['bfr']['method']
        print("Algo: ", algo_name)
        outpath_local_field = os.path.join(outpath, pipeline_id, f"{algo}/Sepia")
        os.makedirs(outpath_local_field, exist_ok=True)

        eng.python_wrapper(fieldmap_path, "", "", header_path , algo_name, outpath_local_field, mask_path, bgfr_params, nargout = 0)
        print("Local Field Created! Calculate metrics and update parameters!")

def di_comp(localfield_path, header_path, mask_path, pipeline_id, outpath):

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
    
    pipelines = dictionary_of_pipelines()

    if pipeline_id  not in pipelines:
        raise ValueError(f"Pipeline {pipeline_id} not found on dictionary: /n {pipelines}")

    for algo in pipelines[pipeline_id]:
        print("Beginning DI algo comp automated running")
        print("Input algo: ", algo)
        di_params = dict_of_algo_params(algo, "di")
        algo_name =  di_params['qsm']['method']
        print("Algo: ", algo_name)
        outpath_chimap = os.path.join(outpath, pipeline_id, f"{algo}/Sepia")
        os.makedirs(outpath_chimap, exist_ok=True)

        eng.python_wrapper(localfield_path, "", "", header_path , algo_name, outpath_chimap, mask_path, di_params, nargout = 0)
        print("Local Field Created! Calculate metrics and update parameters!")



