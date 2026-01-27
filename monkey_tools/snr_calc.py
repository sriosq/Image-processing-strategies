import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt

def snr_calc(img_path, signal_msk_path, noise_msk_path):
    img_data = nib.load(img_path).get_fdata()
    signal_msk_data = nib.load(signal_msk_path).get_fdata()
    noise_msk_data = nib.load(noise_msk_path).get_fdata()
    # We always want just the 1st echo to be covered in the noise_mask
    # If multi-echo, reduce noise mask to first echo (3D)
    if noise_msk_data.ndim == 4:
        noise_msk_data = noise_msk_data[..., 0]
    # Same thing for signal mask
    if signal_msk_data.ndim == 4:
        signal_msk_data = signal_msk_data[..., 0]
    # Check single-echo or multi-echo

    if img_data.ndim == 3:

        signal_mean = np.mean(img_data[signal_msk_data==1])
        noise_std = np.std(img_data[noise_msk_data==1]) * 0.665  # Rician correction factor
        snr = signal_mean / (noise_std ) # Rician correction factor
        return snr, noise_std
    
    if img_data.ndim == 4:

        snr_list = [] # Sorted through echoes
        # Noise should change through echoes!
        # Calculate sigma outside of the echo train loop
        sigma = np.std(img_data[...,0][noise_msk_data == 1]) * 0.665 # Rician correction factor

        for echo_idx in range(img_data.shape[3]):
            signal_mean = np.mean(img_data[..., echo_idx][signal_msk_data==1]) # or "mu"
            snr_list.append(signal_mean / sigma)

        return np.array(snr_list), sigma
    
    else:
        raise ValueError("Image data has invalid number of dimensions - check input image")
