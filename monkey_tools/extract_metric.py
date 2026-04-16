
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import pandas as pd

# This script automates metric extraction using spinal cord toolbox!
# This helps creating the folders when they are not existant and also cleans up the final jupyternotebook + modular baby!

# Important librabry!
import subprocess

def extract_metrics(path_to_local_field, path_to_mask, path_to_vert_levels, path_to_output, method, vert, perlevel="1"):
    """
    Function to extract metrics using a terminal command.

    Parameters:
    - path_to_demod_b0 (str): Path to the demodulated B0 file.
    - path_to_sc_seg (str): Path to the spinal cord segmentation file.
    - path_to_vert_levels (str): Path to the vertebral levels file.
    - path_to_output (str): Path to the output CSV file.
    - method (str): Method used in extraction (default is "wa").
    - vert (str): Vertebral levels to process (default is "2:14").
    - perlevel (str): Per-level processing flag (default is "1").
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(path_to_output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Construct the command
    command = [
        "sct_extract_metric",
        "-i", path_to_local_field,
        "-f", path_to_mask,
        "-method", method,
        "-vert", vert,
        "-vertfile", path_to_vert_levels,
        "-perlevel", perlevel,
        "-o", path_to_output
    ]

    # Run the command and capture output
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print("Error occurred:", result.stderr)
    else:
        print("Metrics extracted successfully:", result.stdout)


# Example usage
#path_to_demod_b0 = "C:/Users/User/msc_project/Image-processing-strategies/compare_fieldmap/data/lung_fitting_project/manually_simulated/demodulated/fftshift_bug/demod_fftshift_ISMRM_img.nii.gz"
#path_to_sc_seg = "D:/UNF_data/2024_08_23/slicer_work/t1w_wholebody_sc_complete.nii.gz"
#path_to_vert_levels = "D:/UNF_data/2024_08_23/slicer_work/t1w_wholebody_label_vertebrae_final.nii.gz"
#path_to_output = "C:/Users/User/msc_project/Image-processing-strategies/compare_fieldmap/data/lung_fitting_project/manually_simulated/extract_metrics/test/simulated_db0_030_metrics.csv"

# Call the function
#extract_metrics(path_to_demod_b0, path_to_sc_seg, path_to_vert_levels, path_to_output, method='wa', vert="2:14")


# Extract the WM and GM values for each dataset
def extract_values_per_vertebrae(input_data, mask, vertfile, participant_id, tissue, statistic="mean"):
    '''
    Extract values per vertebrae level for a given tissue type

    Parameters:
    input_data (numpy.ndarray): Input data from which to extract values (e.g., local field data)
    mask (numpy.ndarray): Binary mask for the region of interest to where values are calculated (e.g., WM or GM) 
    vertfile (numpy.ndarray): Vertebrae level mask to separate values by vertebrae
    participant_id (str): Participant identifier for record purposes
    tissue (str): Tissue type for record purposes (e.g., 'WM' or 'GM')
    
    '''

    records = []
    vertebrae = np.unique(vertfile) # Get the number of verebrae
    vertebrae = vertebrae[vertebrae > 0]  # remove background

    for v in vertebrae:
        #print(f"Calculating for vertebrae: {v}")
        idx = (vertfile == v) & (mask > 0) # Create the indices for the specific vertebrae
        values = input_data[idx]

        if values.size == 0:
            continue  # Skip if no values found for this vertebrae
        std = np.std(values)
        # Caclulate how many voxels are not 0 inside of the mask
        nonzero_voxels = np.count_nonzero(values)

        if statistic == "mean":
            summary_value = np.mean(values)
            #print(f"Mean value for vertebrae {v}, tissue {tissue}: {summary_value}")
        
        elif statistic == "median":
            summary_value = np.median(values)
            #print(f"Median value for vertebrae {v}, tissue {tissue}: {summary_value}")
        else:
            raise ValueError("Statistic not recognized. Use 'mean' or 'median'.")

        records.append({
            'subject': participant_id,
            'vertebrae': int(v),
            'tissue': tissue,
            'metric': summary_value,
            "std": std,
            'tot_voxels': values.size,
            "nozero_voxels": nonzero_voxels
        })

    #print(f"Total records extracted for {tissue}: {len(records)}")

    return records

def multi_algo_comp_wm_gm_custom(dub, meas, algo_type, root_dir, test_folders, gm_msk_path, wm_msk_path, special = 0):
    compare_chimap_rows = []

    for algo in test_folders:

        if algo_type == "bgfr":
            map_path = os.path.join(root_dir, algo, "Sepia_localfield.nii.gz")
        if algo_type == "chi_map":
            map_path = os.path.join(root_dir, algo, "Sepia_Chimap.nii.gz")

        if map_path is None or not os.path.exists(map_path):
            print(f"{algo_type} resulting map not found for {algo}, skipping...")
            continue
        
        map_img = nib.load(map_path)
        map_data = map_img.get_fdata(map_data)
        
        gm_msk_data = nib.load(gm_msk_path).get_fdata()
        wm_msk_data = nib.load(wm_msk_path).get_fdata()

        # Calculate total voxel count
        total_vox_gm = np.sum(gm_msk_data == 1)
        total_vox_wm = np.sum(wm_msk_data == 1)

        wrong_gm_count = 100 * np.count_nonzero((gm_msk_data == 1)&(map_data<0))/total_vox_gm
        wrong_wm_count = 100 * np.count_nonzero((wm_msk_data == 1)&(map_data>0))/total_vox_wm

        # Calculate how many voxels are in each tissue mask
        gm_nonzero_vox = np.sum(map_data[gm_msk_data==1] != 0)
        wm_nonzero_vox = np.sum(map_data[wm_msk_data==1] != 0)

        gm_vals = map_data[(gm_msk_data == 1 )]
        wm_vals = map_data[(wm_msk_data == 1 )]

            # Compute metrics for GM and WM
        gm_mean = np.mean(gm_vals)
        gm_std = np.std(map_data[gm_msk_data == 1])
        
        wm_mean = np.mean(wm_vals)
        wm_std = np.std(map_data[wm_msk_data == 1])
        


        abs_contrast = np.abs(gm_mean - wm_mean) # Subtract wm from gm because wm should be negative
        norm_contrast = abs_contrast / (np.abs(gm_mean) + np.abs(wm_mean))
            # This way, if WM is not negative we get a negative contrast which is bad!

        denominator = np.sqrt((gm_std**2 + wm_std**2)/2)

        raw_metric = abs_contrast / denominator if denominator != 0 else 0 # Just in case that the std is 0 - to avoid division by zero

        gm_penalty = gm_nonzero_vox / total_vox_gm if total_vox_gm != 0 else 0
        wm_penalty = wm_nonzero_vox / total_vox_wm if total_vox_wm != 0 else 0

        # Final metric
        final_metric = raw_metric * gm_penalty * wm_penalty 

            # Additionally, we want to calculate a new metric where we calculate how many voxels are misrepresented
            # We want to see how many voxels in either tissue have the opposite value
        
        # Now collect row and add to data frame
        compare_chimap_rows.append({
                'subject': dub,
                'measurement': meas,
                'test_folder': algo,

                'mean_gm': gm_mean,
                'std_gm': gm_std,
                'total_vox_gm': total_vox_gm,
                'nonzero_vox_gm': gm_nonzero_vox,
                'gm_penality': gm_penalty,

                'mean_wm': wm_mean,
                'std_wm': wm_std,
                'total_vox_wm': total_vox_wm,
                'nonzero_vox_wm': wm_nonzero_vox,
                'wm_penality': wm_penalty,

                'wrong_gm_fraction': wrong_gm_count,
                'wrong_wm_fraction': wrong_wm_count,
    
                'contrast_factor': abs_contrast,
                'contrast_percent': norm_contrast,

                'std_denominator': denominator,
                'raw_metric': raw_metric,
                
                'final_metric': final_metric
                })
        
        df = pd.DataFrame(compare_chimap_rows)
    
    if special:
        final_metric_sorted_df = df.sort_values(by='final_metric', ascending=False)
        contrast_sorted_df = df.sort_values(by='contrast_percent', ascending=False)
        noise_sorted_df = df.sort_values(by='std_denominator', ascending=True)
        return final_metric_sorted_df, contrast_sorted_df, noise_sorted_df, df
    else: 
        return df

# Now similar but for just 1 algo, compare along slices

def single_algo_comp(dub, meas, algo_type, root_dir, test_folders, sc_msk_path, gm_msk_path, wm_msk_path, vertfile=None):
    compare_chimap_rows = []

    # This function assumes that the test_folders only contains the same algo but different parameters
    # So we can compare the metrics along the vertebra levels instead of combining in to a whole mask metric
    for algo in test_folders:
        map_path = None
        if algo_type == "bgfr":
            map_path = os.path.join(root_dir, algo, "Sepia_localfield.nii.gz")
        elif algo_type == "chi_map":
            map_path = os.path.join(root_dir, algo, "Sepia_Chimap.nii.gz")

        elif map_path is None or not os.path.exists(map_path):
            print(f"Map not found for {algo}, skipping...")
            continue
        
        map_img = nib.load(map_path)
        map_data = map_img.get_fdata()

        sc_msk_data = nib.load(sc_msk_path).get_fdata()
        gm_msk_data = nib.load(gm_msk_path).get_fdata()
        wm_msk_data = nib.load(wm_msk_path).get_fdata()

        # Now loop through slices

        if vertfile:
            # This means we want to calculate GM/WM values per vertebrae and create a DF with vertebrae values per participant and per algo
            vertfile_img = nib.load(vertfile)
            vertfile_data = vertfile_img.get_fdata()

            # We already hav the map data and the extract vert values function
            gm_mean_df = pd.DataFrame(extract_values_per_vertebrae(map_data, gm_msk_data, vertfile_data, participant_id=dub, tissue="GM", statistic="mean"))
            wm_mean_df = pd.DataFrame(extract_values_per_vertebrae(map_data, wm_msk_data, vertfile_data, participant_id=dub, tissue="WM", statistic="mean"))

            gm_median_df = pd.DataFrame(extract_values_per_vertebrae(map_data, gm_msk_data, vertfile_data, participant_id=dub, tissue="GM", statistic="median"))
            wm_median_df = pd.DataFrame(extract_values_per_vertebrae(map_data, wm_msk_data, vertfile_data, participant_id=dub, tissue="WM", statistic="median"))
            # Now, we create the DF with a column for each vert level and calculate the contrast, cnr and weighted cnr for each vert level for GM and WM together
            vertl_lvl_df = pd.DataFrame()
            
            vertebrae = np.unique(vertfile_data)
            vertebrae = vertebrae[vertebrae > 0]
             
            for v in vertebrae:
                gm_mean = gm_mean_df.loc[gm_mean_df["vertebrae"] == v, "metric"].values[0]
                gm_std = gm_mean_df.loc[gm_mean_df["vertebrae"] == v, "std"].values[0]
                gm_median = gm_median_df.loc[gm_median_df["vertebrae"] == v, "metric"].values
                gm_tot_vox = gm_mean_df.loc[gm_mean_df["vertebrae"] == v, "tot_voxels"].values[0]
                gm_nonzero_vox = gm_mean_df.loc[gm_mean_df["vertebrae"] == v, "nozero_voxels"].values[0]


                gm_penalty = gm_mean_df.loc[gm_mean_df["vertebrae"] == v, "nozero_voxels"].values[0] / gm_mean_df.loc[gm_mean_df["vertebrae"] == v, "tot_voxels"].values[0] if gm_mean_df.loc[gm_mean_df["vertebrae"] == v, "tot_voxels"].values[0] != 0 else 0
                wrong_gm_count = 100 * np.count_nonzero(
                (gm_msk_data == 1) & (map_data < 0) & (vertfile_data == v)
                ) / gm_tot_vox if gm_tot_vox != 0 else np.nan

                wm_mean = wm_mean_df.loc[wm_mean_df["vertebrae"] == v, "metric"].values[0]
                wm_std = wm_mean_df.loc[wm_mean_df["vertebrae"] == v, "std"].values[0]
                wm_median = wm_median_df.loc[wm_median_df["vertebrae"] == v, "metric"].values
                wm_tot_vox = wm_mean_df.loc[wm_mean_df["vertebrae"] == v, "tot_voxels"].values[0]
                wm_nonzero_vox = wm_mean_df.loc[wm_mean_df["vertebrae"] == v, "nozero_voxels"].values[0]


                wm_penalty = wm_mean_df.loc[wm_mean_df["vertebrae"] == v, "nozero_voxels"].values[0] / wm_mean_df.loc[wm_mean_df["vertebrae"] == v, "tot_voxels"].values[0] if wm_mean_df.loc[wm_mean_df["vertebrae"] == v, "tot_voxels"].values[0] != 0 else 0
                wrong_wm_count = 100 * np.count_nonzero(
                (wm_msk_data == 1) & (map_data > 0) & (vertfile_data == v)
                ) / wm_tot_vox if wm_tot_vox != 0 else np.nan

                # Now, use the calculated metrics to calculate new more meaningful metrics
                abs_contrast = np.abs(gm_mean - wm_mean)

                contrast_denominator = np.abs(gm_mean) + np.abs(wm_mean)
                invalid_contrast_percent = contrast_denominator <= 1e-12

                if invalid_contrast_percent:
                    print(
                        f"[INVALID contrast_percent] subject={dub}, measurement={meas}, "
                        f"algo={algo}, vertebra={int(v)}, "
                        f"gm_mean={gm_mean}, wm_mean={wm_mean}, "
                        f"denominator={contrast_denominator}"
                    )
                    norm_contrast = np.nan
                else:
                    norm_contrast = abs_contrast / contrast_denominator

                # This way, if WM is not negative we get a negative contrast which is bad!

                std_denominator = np.sqrt((gm_std**2 + wm_std**2)/2)
                raw_metric = abs_contrast / std_denominator if std_denominator != 0 else 0 # Just in case that the std is 0 - to avoid division by zero
                final_metric = raw_metric * gm_penalty * wm_penalty

                compare_chimap_rows.append({
                    'subject': dub,
                    'measurement': meas,
                    'test_folder': algo,
                    'vertebrae': int(v),

                    'mean_gm': gm_mean,
                    'median_gm': gm_median,
                    'std_gm': gm_std,
                    'total_vox_gm': gm_tot_vox,
                    'nonzero_vox_gm': gm_nonzero_vox,
                    'gm_penality': gm_penalty,

                    'mean_wm': wm_mean,
                    "median_wm": wm_median,
                    'std_wm': wm_std,
                    'total_vox_wm': wm_tot_vox,
                    'nonzero_vox_wm': wm_nonzero_vox,
                    'wm_penality': wm_penalty,

                    'wrong_gm_fraction': wrong_gm_count,
                    'wrong_wm_fraction': wrong_wm_count,
        
                    'contrast_factor': abs_contrast,
                    'contrast_percent': norm_contrast,

                    'contrast_percent_denominator': contrast_denominator,
                    'invalid_contrast_percent': invalid_contrast_percent,

                    'std_denominator': std_denominator,
                    'raw_metric': raw_metric,
                    
                    'final_metric': final_metric
                    })

        else: # We do slicewise if no vertfile is provided
            for slice_idx in range(map_data.shape[2]): # Go through z direction
                map_slice = map_data[:, :, slice_idx]
                sc_msk_slice = sc_msk_data[:, :, slice_idx]
                gm_msk_slice = gm_msk_data[:, :, slice_idx]
                wm_msk_slice = wm_msk_data[:, :, slice_idx]

                total_slice_vox_gm = np.sum(gm_msk_slice == 1)
                total_slice_vox_wm = np.sum(wm_msk_slice == 1)

                gm_vals = map_slice[(gm_msk_slice == 1 ) & (map_slice != 0)].ravel()
                wm_vals = map_slice[(wm_msk_slice == 1 ) & (map_slice != 0)].ravel()

                if gm_vals.size == 0 or wm_vals.size == 0:
                    continue  # Skip if no values found for this slice

                # Compute metrics for GM and WM
                gm_mean = np.mean(gm_vals)
                gm_std = np.std(map_slice[gm_msk_slice == 1])
                
                wm_mean = np.mean(wm_vals)
                wm_std = np.std(map_slice[wm_msk_slice == 1])

                sc_std = np.std(map_slice[sc_msk_slice == 1])
                
                contrast = np.abs(gm_mean - wm_mean) # Subtract wm from gm because wm should be negative
                # This way, if WM is not negative we get a negative contrast which is bad!

                #denominator = np.sqrt((gm_std**2 + wm_std**2)/2)
                denominator = np.sqrt(sc_std**2/2)

                # We expect that gm_mean to be 0.03 and wm_mean to be -0.03 
                # therefore contrast should be 0.06 ideally! we will penalized based on how far we are
                gm_metric_penalty = np.abs(gm_mean - 0.03) / 0.03
                wm_metric_penalty = np.abs(wm_mean + 0.03) / 0.03

                penalized_contrast = np.abs(gm_metric_penalty - wm_metric_penalty)
                contrast = penalized_contrast
                raw_metric = contrast / denominator if denominator != 0 else 0 # Just in case that the std is 0 - to avoid division by zero

                gm_nonzero_vox = np.sum(map_slice[gm_msk_slice==1] != 0)
                wm_nonzero_vox = np.sum(map_slice[wm_msk_slice==1] != 0)

                # Now collect row and add to data frame
                gm_penalty = gm_nonzero_vox / total_slice_vox_gm if total_slice_vox_gm != 0 else 0
                wm_penalty = wm_nonzero_vox / total_slice_vox_wm if total_slice_vox_wm != 0 else 0

                # Final metric
                final_metric = raw_metric * gm_penalty * wm_penalty 

                        # Now collect row and add to data frame
                compare_chimap_rows.append({
                    'subject': dub,
                    'measurement': meas,
                    'test_folder': algo,
                    'slice_index': slice_idx,
                    'mean_gm': gm_mean,
                    'std_gm': gm_std,
                    'total_vox_gm': total_slice_vox_gm,
                    'nonzero_vox_gm': gm_nonzero_vox,
                    'gm_penality': gm_penalty,

                    'mean_wm': wm_mean,
                    'std_wm': wm_std,
                    'total_vox_wm': total_slice_vox_wm,
                    'nonzero_vox_wm': wm_nonzero_vox,
                    'wm_penality': wm_penalty,
        
                    'contrast_factor': contrast,
                    'std_denominator': denominator,
                    'raw_metric': raw_metric,
                    
                    'final_metric': final_metric
                    })
                         
    # Either vert level metrics or slice wise metrics, we create the DF at the end of the algo loop    
    df = pd.DataFrame(compare_chimap_rows)
    return df


def calculate_rmse(input_map, gt_map, mask, gm_msk, wm_msk):
    '''
    Calculate RMSE between input map and ground truth map within a specified mask.

    Parameters:
    input_map (numpy.ndarray): The input map for which RMSE is to be calculated.
    gt_map (numpy.ndarray): The ground truth map to compare against.
    mask (numpy.ndarray): A binary mask specifying the region of interest for RMSE calculation.

    Returns:
    float: The calculated RMSE value.
    '''

    pixel_wise_difference = gt_map - input_map
    gm_diff = pixel_wise_difference[gm_msk==1]
    wm_diff = pixel_wise_difference[wm_msk==1]

    gm_mean_diff = np.mean(gm_diff)
    gm_std_diff = np.std(gm_diff)
    gm_rmse = np.sqrt(np.mean(gm_diff ** 2))

    wm_mean_diff = np.mean(wm_diff)
    wm_std_diff = np.std(wm_diff)
    wm_rmse = np.sqrt(np.mean(wm_diff ** 2))

    global_rmse = np.sqrt(np.mean(pixel_wise_difference[mask==1] ** 2))

    return gm_rmse, wm_rmse, global_rmse
