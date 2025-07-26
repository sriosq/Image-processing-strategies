import nibabel as nib

# Small function that recieves a FM img in PPM and returns in Hz using specified B0
def convert_img_ppm_to_Hz_data(nifti_ppm, B0):
    fm_ppm = nifti_ppm.get_fdata()
    # When creating fieldmaps with susceptiblity to fieldmap repository, the output will be in PPM so we need to rescale to Tesla
    if B0 != None:
        gamma_bar = 42.58 # [Hz/T]
        f0 = B0*gamma_bar
        fm_Hz = fm_ppm*f0
        return fm_Hz
    else:
        print("No B0 strength give please especify")
        return ValueError("B0 must be different than None")
    
def convert_ppm_data_to_Hz_data(ppm_fm, B0):
    if B0 != None:
        gamma_bar = 42.58 # [Hz/T]
        f0 = B0*gamma_bar
        fm_Hz = ppm_fm*f0
        return fm_Hz
    else:
        print("No B0 strength give please especify")
        return ValueError("B0 must be different than None")
    



    
