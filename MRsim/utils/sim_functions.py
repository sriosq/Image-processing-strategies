import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from math import pi

def optimize_measurement(pd_vol, t2s_vol, dims, deltaB0, FA,TE,B0):
    # This code seeks to accomplish the same as the above method but
    # We are trying to optimize by using volumes
    # TE should be a list, so we create a new volume
    print("Starting optimize_measurement")
    num_TE = len(TE)
    newVol_dims = list(dims)
    newVol_dims.append(num_TE)
    # This way we can iterate over the last dimension (TEs)

    magnitude = np.zeros(newVol_dims)
    phase = np.zeros(newVol_dims)

    gamma = 42.58 * B0  * 2 * pi  # This is rad*Hz/Tesla 
    gamma_rd_sT = 267.52218744 * 1e6 # In rad/(sec * T) it needs to be 10e5 or 1e6 lol
    handedness = 'left'

    for te_idx, TE_val in enumerate(TE):
        print(f"Processing TE[{te_idx}] = {TE_val}")
        mag, phase_arr = optimized_signal3(pd_vol, t2s_vol, FA, TE_val, deltaB0, gamma_rd_sT, B0, handedness)
        print(f"mag shape: {mag.shape}, phase_arr shape: {phase_arr.shape}")
        if mag.shape != tuple(dims):
            raise ValueError(f"Shape mismatch: expected {tuple(dims)} but got {mag.shape}")
        magnitude[..., te_idx] = mag
        phase[..., te_idx] = phase_arr
       
    print("Finished optimize_measurement")

    return magnitude, phase

def optimized_signal(pd_vol,T2star_vol, FA, te, deltaB0_vol, gamma, handedness):
    # This is an optimized version from generate_signal, using numpy array matrices
    print("Starting optimized_signal")
    decay = np.exp(-te / (T2star_vol * 1e-3))  # Convert T2s values to same as Echo Times and apply decay exponential
    phase_factor = -1j * gamma * deltaB0_vol * te if handedness == 'left' else 1j * gamma * deltaB0_vol * te
    # Phase factor in radians

    signal = pd_vol * np.sin(FA) * decay * np.exp(phase_factor)
    print("Finished optimized_signal")
    return np.abs(signal), np.angle(signal) # Abs for the Magnitude whereas angle for Phase

def optimized_signal2(pd_vol,T2star_vol, FA, te, deltaB0_vol, gamma, handedness):
    # This is an optimized version from generate_signal, using numpy array matrices
    print("Starting optimized_signal")
    decay = np.exp(-te  / (T2star_vol * 1e-3))  # Convert T2s values to same as Echo Times and apply decay exponential
    if handedness == 'left':
        print('handedness=left')
        phase_factor = -1j * gamma * deltaB0_vol * te* 1e-3   
    elif handedness == 'right':
        print('handedness=right')
        1j * gamma * deltaB0_vol * te* 1e-3 
    else:
        print('wrong handedness')
    # Phase factor in radians

    signal = pd_vol * np.sin(FA) * decay * np.exp(phase_factor)
    print("Finished optimized_signal")
    return np.abs(signal), np.angle(signal) # Abs for the Magnitude whereas angle for Phase

######################### ---------------------------------- ################################
# FINAL VERSION IS THIS:
# The units of the phase factor are only correct in this version
def optimized_signal3(pd_vol,T2star_vol, FA, te, deltaB0_vol, gamma, B0, handedness):
    # This is an optimized version from generate_signal, using numpy array matrices
    print("Starting optimized_signal")
    decay = np.exp(-te  / (T2star_vol * 1e-3))  # Convert T2s values to same as Echo Times and apply decay exponential
    fa = np.deg2rad(FA)
    print("sin: ", np.sin(fa))
    
    if handedness == 'left':

        print('handedness=left')
        phase_factor = -1j * gamma * deltaB0_vol * B0 * 1e-6 * te
        coef = -1j * gamma * B0 * 1e-6 * te
        print("Coefficient of phase factor: ", coef)

    elif handedness == 'right':

        print('handedness=right')
        phase_factor =  1j * gamma * deltaB0_vol * B0* 1e-6 * te
        coef = 1j * gamma * B0 * 1e-6 * te
        print("Coefficient of phase factor: ", coef)

    else:
        print('wrong handedness')
    # Phase factor in radians

    signal = pd_vol * np.sin(fa) * decay * np.exp(phase_factor)
    print("Finished optimized_signal")

    return np.abs(signal), np.angle(signal) # Abs for the Magnitude whereas angle for Phase