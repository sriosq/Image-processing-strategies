import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt

def snr_calc(img_path, signal_msk_path, noise_msk_path):
    img_data = nib.load(img_path).get_fdata()
    signal_msk_data = nib.load(signal_msk_path).get_fdata()
    noise_msk_data = nib.load(noise_msk_path).get_fdata()

    signal_mean = np.mean(img_data[signal_msk_data==1])
    noise_std = np.std(img_data[noise_msk_data==1])

    snr = signal_mean / noise_std
    return snr
