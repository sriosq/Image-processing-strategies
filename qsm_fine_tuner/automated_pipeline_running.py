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
    }
    return pipelines

def signal_to_fm_Hz(mag_path, ph_path, TEs, outfn, msk_path=None, romeo_path="c:/romeo/romeo.jl"):

    TE_str = "[" + ", ".join(str(TE) for TE in TEs) + "]"

    # Build command as a LIST, not a string
    cmd = ["julia", romeo_path, "-p", ph_path, "-m", mag_path]

    if msk_path is not None:
        cmd += ["-k", msk_path, "-B", "-Q", "-t", TE_str, "-u", "-o", outfn]
    else:
        cmd += ["-B", "-Q", "-t", TE_str, "-o", outfn]

    print("Running ROMEO with command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print(f"ROMEO completed. Output saved in: {outfn}")

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


