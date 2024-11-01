import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import pandas as pd


def demod_pls(image_data, mask_data):
    print("Demod function activated")
    # Calculate the mean on the mask
    demod_factor = np.mean(image_data[mask_data == 1])
    demod_img_data = image_data - demod_factor
    print("Demodulation with: ",demod_factor," Hz")
    return demod_img_data



