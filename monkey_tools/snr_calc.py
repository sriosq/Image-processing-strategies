import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt

def snr_calc(img_path, signal_msk_path, noise_msk_path, verbose=0, correct_rician=0):
    '''
    Function to calculate SNR provided with 3 paths
    - img_path = Path to nifti image (Can be 3D or 4D)
    - signal_msk_path = Path to the mask where the signal intensity will be calculated (If 4D, will use 1st volume)
    - noise_msk_path = Path to mask on the BG where std is calculated (If 4D, will use 1st volume)
    
    Optional arguments:
    - verbose = By default off, if enabled will print Mean and Noise std
    - correct_rician = Magnitude and phase operation is nonlinear therefore change nature of noise from Gaussian to rician in the BG
    '''


    img_data = nib.load(img_path).get_fdata()
    signal_msk_data = nib.load(signal_msk_path).get_fdata()
    noise_msk_data = nib.load(noise_msk_path).get_fdata()
    # We always want just the 1st echo to be covered in the noise_mask
    # If multi-echo, reduce noise mask to first echo (3D)
    if noise_msk_data.ndim > 3:
        noise_msk_data = noise_msk_data[..., 0]
    # Same thing for signal mask
    if signal_msk_data.ndim > 3:
        signal_msk_data = signal_msk_data[..., 0]
    # Check single-echo or multi-echo

    if img_data.ndim == 3:

        signal_mean = np.mean(img_data[signal_msk_data==1])
        print("Signal mean",signal_mean)
        if correct_rician:
            noise_std = np.std(img_data[noise_msk_data==1]) / 0.665  # Rician correction factor
            if verbose: print("Rician corrected noise: ",noise_std) 
        else:
            noise_std = np.std(img_data[noise_msk_data==1])
            if verbose: print("Noise std: ",noise_std)

            
        snr = signal_mean / noise_std
        
        return snr, noise_std
    
    if img_data.ndim == 4:

        snr_list = [] # Sorted through echoes
        # SNR should change through echoes!
        # But noise on the BG should not!
        # Calculate sigma outside of the echo loop
        if correct_rician:
            sigma = np.std(img_data[..., 0][noise_msk_data == 1]) / 0.665 # Rician correction factor
            if verbose: print("Rician corrected noise: ", sigma) 
        else:
            sigma = np.std(img_data[..., 0][noise_msk_data == 1])
            if verbose: print("Noise std: ",noise_std)

        for echo_idx in range(img_data.shape[3]):
            signal_mean = np.mean(img_data[..., echo_idx][signal_msk_data==1]) # or "mu"
            snr_list.append(signal_mean / sigma)
            if verbose:
                print("Signal mean",signal_mean)

        return np.array(snr_list), sigma
    
    else:
        raise ValueError("Image data has invalid number of dimensions - check input image")
