import os 
import nibabel as nib
import numpy as np
import pandas as pd

def compare_algo_metrics_from_folder(dubs, measurements, folder_path, algo_type, testing_subfodlers, gm_msk_path, wm_msk_path):

    '''
    Compare algorithms based on metrics computed from images stored in a folder structure.
    Parameters:
    - dubs: List of subject identifiers.
    - measurements: List of measurement identifiers.
    - folder_path: Path to the main folder containing algorithm subfolders.
    - algo_type: Type of algorithm ('Chi_mapping' or 'BGFR').
    - testing_subfodlers: List of subfolder names corresponding to different algorithms.
    - gm_msk_path: Path to the gray matter mask file.
    - wm_msk_path: Path to the white matter mask file.
    Returns:
    - algo_comp_df: DataFrame containing computed metrics for each algorithm.
    '''

    df_rows = []

    if algo_type == "Chi_mapping":
            image_fn = 'Sepia_Chimap.nii.gz'
    elif algo_type == "BGFR":
            image_fn = 'Sepia_localfield.nii.gz'

    for dub in dubs:
        for meas in measurements:
            for algo in testing_subfodlers:
            
            # Firsst, define the path pointers
            
                map_path = os.path.join(folder_path, algo, image_fn)
                if map_path is None or not os.path.exists(map_path):
                    print(f"Skipping {map_path} as it does not exist.")
                    continue
                
                #gm_msk_path = os.path.join(subj_meas_path, f'custom_{subj}_{meas}_gm_msk.nii.gz')
                #wm_msk_path = os.path.join(subj_meas_path, f'custom_{subj}_{meas}_wm_msk.nii.gz')

                # Load data
                map_img = nib.load(map_path)
                map_data = map_img.get_fdata()

                gm_vals = map_data[(gm_mask == 1) & (map_data != 0)].ravel()
                wm_vals = map_data[(wm_mask == 1) & (map_data != 0)].ravel()

                gm_mask = nib.load(gm_msk_path).get_fdata()
                total_vox_gm = np.sum(gm_mask==1) 
                wm_mask = nib.load(wm_msk_path).get_fdata()
                total_vox_wm = np.sum(wm_mask==1)

                # Compute metrics for GM and WM
                gm_mean = np.mean(map_data[gm_mask==1])
                gm_std = np.std(map_data[gm_mask==1])
                n_gm = gm_mean.size
                wm_mean = np.mean(map_data[wm_mask==1])
                wm_std = np.std(map_data[wm_mask==1])
                n_wm = wm_mean.size

                # Compute how many voxels are in the mask
                gm_nonzero_vox = np.sum(map_data[gm_mask==1] != 0)
                wm_nonzero_vox = np.sum(map_data[wm_mask==1] != 0)

                # Compute the contrast metric
                # We want to maximize the contrast between GM and WM
                # WM mean should be negative, if it is positive the contrast will be lower this way
                contrast = np.abs(gm_mean - wm_mean)

                normalizer_denominator_2 = np.sqrt(gm_std**2 + wm_std**2) 
            
                raw_metric = contrast / normalizer_denominator_2 if normalizer_denominator_2 != 0 else 0 # Just in case that the std is 0 - to avoid division by zero

                # Now we penalize if the algo eroded the mask
                gm_penality = gm_nonzero_vox / total_vox_gm 
                wm_penality = wm_nonzero_vox / total_vox_wm

                # Final metric:
                final_metric = raw_metric * gm_penality * wm_penality

                # Now collect row and add to data frame
                df_rows.append({
                        
                    'subject': 'hc2',
                    'measurement': 'm1',
                    'test_folder': algo,

                    'mean_gm': gm_mean,
                    'std_gm': gm_std,
                    'total_vox_gm': total_vox_gm,
                    'nonzero_vox_gm': gm_nonzero_vox,
                    'gm_penality': gm_penality,

                    'mean_wm': wm_mean,
                    'std_wm': wm_std,
                    'total_vox_wm': total_vox_wm,
                    'nonzero_vox_wm': wm_nonzero_vox,
                    'wm_penality': wm_penality,
        
                    'contrast_factor': contrast,
                    'pooled_std_denominator': normalizer_denominator_2,
                    #'raw_metric': raw_metric,
                    
                    'final_metric': final_metric
                })


    # Create the data frame
    algo_comp_df = pd.DataFrame(df_rows)

    return algo_comp_df

def calculate_spine_avg_qsm_metrics(root, dubs, measurements, algo_identifier):
     
    spine_gm_avg = []
    spine_wm_avg = []

    for dub in dubs:
        for meas in measurements:
            
            gm_metrics_path = os.path.join(root, dub, meas, f"{dub}{meas}_{algo_identifier}_gm_metrics.csv")
            wm_metrics_path = os.path.join(root, dub, meas, f"{dub}{meas}_{algo_identifier}_wm_metrics.csv")

            gm_metrics_df = pd.read_csv(gm_metrics_path)
            wm_metrics_df = pd.read_csv(wm_metrics_path)        

            gm_avg = np.mean(gm_metrics_df['WA()'])
            wm_avg = np.mean(wm_metrics_df['WA()'])

            spine_gm_avg.append(gm_avg)
            spine_wm_avg.append(wm_avg)

    return spine_gm_avg, spine_wm_avg

def calculate_t_statistic_between_gm_wm(chimap_path, gm_msk_path, wm_mask_path):
    from scipy.stats import ttest_ind
    chimap_data = nib.load(chimap_path).get_fdata()
    wm_mask_data = nib.load(wm_mask_path).get_fdata()   
    gm_mask_data = nib.load(gm_msk_path).get_fdata()

    wm_vals = chimap_data[wm_mask_data == 1]
    gm_vals = chimap_data[gm_mask_data == 1]

# Remove NaN/inf values
    wm_vals = wm_vals[np.isfinite(wm_vals)]
    gm_vals = gm_vals[np.isfinite(gm_vals)]

# Means
    wm_mean = np.mean(wm_vals)
    gm_mean = np.mean(gm_vals)
    mean_diff = gm_mean - wm_mean  # Substract WM because it should always be negative

    t_stat, p_val = ttest_ind(wm_vals, gm_vals, equal_var=False)
    print(f"T-statistic: {t_stat:.6f}, p-value: {p_val:.6e}")
    
    return t_stat, p_val