import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from math import pi
import os

# The code in this section is highly influenced by Chapter 2.5 of QSM RC 2.0

def complete_measurement(t1_vol, pd_vol, t2s_vol, dims, deltaB0, FA ,TE, TR, B0, per_echo = 0, outpath="", handedness='right'):
    # T1, PD and T2* volumes assumed to have values in [ms]
    # dB0 is the Fieldmap in [ppm]
    # Flip angle input in degrees
    # Echo time can be a single echo in seconds [s] or a list of echo times [s]
    # Repetition time in [ms]
    # B0 is the magnetic field strength in Tesla [T]

    print("Starting optimize_measurement")
    num_TE = len(TE)
    newVol_dims = list(dims)
    newVol_dims.append(num_TE)
    # This way we can iterate over the last dimension (TEs)

    magnitude = np.zeros(newVol_dims)
    phase = np.zeros(newVol_dims)

    # gamma = 42.58 * B0  * 2 * pi  # This is rad*Hz/Tesla 
    gamma_rd_sT = 267.52218744 * 1e6 # In rad/(sec * T) it needs to be 10e5 or 1e6 

    fa = np.deg2rad(FA)
    print("Flip angle used for:")
    print("sin($/alpha$): ", np.sin(fa))
    print("1-cos($/alpha$): ", 1-np.cos(fa))
    
    for te_idx, TE_val in enumerate(TE):
        print(f"Processing TE[{te_idx}] = {TE_val}"," [s]")

        mag, phase_arr = simulation_complete(pd_vol, t2s_vol, t1_vol, fa, TE_val, TR, deltaB0, gamma_rd_sT, B0, handedness)

        print(f"mag shape: {mag.shape}, phase_arr shape: {phase_arr.shape}")
        if mag.shape != tuple(dims):
            raise ValueError(f"Shape mismatch: expected {tuple(dims)} but got {mag.shape}")
        
        magnitude[..., te_idx] = mag
        phase[..., te_idx] = phase_arr

        if per_echo == 1 and outpath != None:
            if outpath.path.exists()==True:
                print("Path already exists, change to avoid overwriting")
            else:
                temp_img = nib.Nifti1Image(mag)

    print("Finished complete measurement")

    return magnitude, phase

def simulation_complete(pd_vol, T2star_vol, T1_vol, fa, te, tr, deltaB0_vol, gamma, B0, handedness):
    # Uses complete equation to simulate Spoiled gradient-recalled-echo data from steady state equation
    print("Using T1, T2* and PD for simulation")

    dims = np.array(pd_vol.shape)
    t2star_decay = np.zeros(dims)

    t2star_decay = np.exp(-te*1e3 / T2star_vol) # te [s] *1000 / T2* [ms]
    phi_zero = 1 # Initial phase distribution from transceiver phase. Set  to zero // exp(i*0) = 1
    # gamma input is in rad/(s*T)
    
    if handedness == 'left':

        print('handedness=left')
        phase_factor = -1j * gamma * deltaB0_vol * 1e-6 * B0 * te 
        coef = -1j * gamma * B0 * 1e-6 * te

        print("Coefficient of phase factor: ", coef)

    elif handedness == 'right':

        print('handedness=right')
        phase_factor =  1j * gamma * deltaB0_vol * 1e-6 * B0  * te 
        coef = 1j * gamma * B0 * 1e-6 * te

        print("Coefficient of phase factor: ", coef)

    else:
        print('wrong handedness')
    # Phase factor in radians

    # Calculation of longitudinal term assuming input flip angle is already in radians
    longitudinal_num = 1 - np.exp(-tr/T1_vol) # [ms]/[ms]
    longitudinal_den = 1 - np.cos(fa)*np.exp(-tr/T1_vol) # [ms]/[ms] 
    longitudinal_term = longitudinal_num/longitudinal_den

    signal = pd_vol * np.sin(fa) * longitudinal_term * t2star_decay * np.exp(phase_factor)  
    # Here we use Proton Density to modulte the signal assuming that M0 is equal to PD 
    # High water content regions have 90 to 100 PD value, whereas bone and air cavities have 20 to 30 PD value
    print("Finished optimized_signal")

    return np.abs(signal), np.angle(signal)