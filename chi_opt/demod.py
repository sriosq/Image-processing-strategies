import numpy as np
import nibabel as nib
import pandas as pd
from utils.extract_metric import extract_metrics

def demod_Hz(path_to_fm, path_to_dmod_msk, outpath = 0):
    raw_iFm_img = nib.load(path_to_fm)
    raw_iFm_data = raw_iFm_img.get_fdata()
    dmod_msk = nib.load(path_to_dmod_msk).get_fdata()

    demod_value = np.mean(raw_iFm_data[dmod_msk == 1])
    print("Demodulating the input fieldmap with a value of: ", demod_value, 'Hz')
    dmod_fm_data = raw_iFm_data - demod_value
    
    
    if outpath != 0:
        print("Saving demodulated fieldmap to nifti!")
        dmod_fm_img = nib.Nifti1Image(dmod_fm_data, affine = raw_iFm_img.affine)
        nib.save(dmod_fm_img, outpath)
        return dmod_fm_data
    
    return dmod_fm_data
