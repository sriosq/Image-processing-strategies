
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
        print(f"Calculating for vertebrae: {v}")
        idx = (vertfile == v) & (mask > 0) # Create the indexes for the specific vertebrae
        values = input_data[idx]

        if values.size == 0:
            continue  # Skip if no values found for this vertebrae

        if statistic == "mean":
            summary_value = np.mean(values)
            print(f"Mean value for vertebrae {v}, tissue {tissue}: {summary_value}")
        
        elif statistic == "median":
            summary_value = np.median(values)
            print(f"Median value for vertebrae {v}, tissue {tissue}: {summary_value}")
        else:
            raise ValueError("Statistic not recognized. Use 'mean' or 'median'.")


        records.append({
            'Participant_ID': participant_id,
            'Vertebrae': int(v),
            'Tissue': tissue,
            'Value': summary_value,
            'Nvoxels': values.size
        })

    print(f"Total records extracted for {tissue}: {len(records)}")

    return records

def multi_algo_comp_wm_gm_custom(dub, meas, algo_type, root_dir, test_folders, gm_msk_path, wm_msk_path):
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
        map_data = map_img.get_fdata()

        gm_msk_data = nib.load(gm_msk_path).get_fdata()
        wm_msk_data = nib.load(wm_msk_path).get_fdata()

        gm_vals = map_data[(gm_msk_data == 1 ) & (map_data != 0)].ravel()
        wm_vals = map_data[(wm_msk_data == 1 ) & (map_data != 0)].ravel()

        # Calculate total voxel count
        total_vox_gm = np.sum(gm_msk_data == 1)
        total_vox_wm = np.sum(wm_msk_data == 1)
        
        # Compute metrics for GM and WM
        gm_mean = np.mean(gm_vals)
        gm_std = np.std(map_data[gm_msk_data == 1])
        
        wm_mean = np.mean(wm_vals)
        wm_std = np.std(map_data[wm_msk_data == 1])
        
        # Calculate how many voxels are in the mask
        gm_nonzero_vox = np.sum(map_data[gm_msk_data==1] != 0)
        wm_nonzero_vox = np.sum(map_data[wm_msk_data==1] != 0)

        contrast = np.abs(gm_mean - wm_mean) # Subtract wm from gm because wm should be negative
        # This way, if WM is not negative we get a negative contrast which is bad!

        denominator = np.sqrt((gm_std**2 + wm_std**2)/2)

        raw_metric = contrast / denominator if denominator != 0 else 0 # Just in case that the std is 0 - to avoid division by zero

        gm_penalty = gm_nonzero_vox / total_vox_gm if total_vox_gm != 0 else 0
        wm_penalty = wm_nonzero_vox / total_vox_wm if total_vox_wm != 0 else 0

        # Final metric
        final_metric = raw_metric * gm_penalty * wm_penalty 

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
    
                'contrast_factor': contrast,
                'std_denominator': denominator,
                'raw_metric': raw_metric,
                
                'final_metric': final_metric
                })
        
        df = pd.DataFrame(compare_chimap_rows)
    return df

# Now similar but for just 1 algo, compare along slices

def single_algo_comp(dub,meas, root_dir, test_folders, sc_msk_path, gm_msk_path, wm_msk_path, metric=None):
    compare_chimap_rows = []

    for algo in test_folders:

        chimap_map_path = os.path.join(root_dir, algo, "Sepia_Chimap.nii.gz")

        if chimap_map_path is None or not os.path.exists(chimap_map_path):
            print(f"Chi map not found for {algo}, skipping...")
            continue
        
        chimap_img = nib.load(chimap_map_path)
        chimap_data = chimap_img.get_fdata()

        sc_msk_data = nib.load(sc_msk_path).get_fdata()
        gm_msk_data = nib.load(gm_msk_path).get_fdata()
        wm_msk_data = nib.load(wm_msk_path).get_fdata()

        # Now loop through slices
        for slice_idx in range(chimap_data.shape[2]): # Go through z direction
            chimap_slice = chimap_data[:, :, slice_idx]
            sc_msk_slice = sc_msk_data[:, :, slice_idx]
            gm_msk_slice = gm_msk_data[:, :, slice_idx]
            wm_msk_slice = wm_msk_data[:, :, slice_idx]

            total_slice_vox_gm = np.sum(gm_msk_slice == 1)
            total_slice_vox_wm = np.sum(wm_msk_slice == 1)

            gm_vals = chimap_slice[(gm_msk_slice == 1 ) & (chimap_slice != 0)].ravel()
            wm_vals = chimap_slice[(wm_msk_slice == 1 ) & (chimap_slice != 0)].ravel()

            if gm_vals.size == 0 or wm_vals.size == 0:
                continue  # Skip if no values found for this slice

            # Compute metrics for GM and WM
            gm_mean = np.mean(gm_vals)
            gm_std = np.std(chimap_slice[gm_msk_slice == 1])
            
            wm_mean = np.mean(wm_vals)
            wm_std = np.std(chimap_slice[wm_msk_slice == 1])

            sc_std = np.std(chimap_slice[sc_msk_slice == 1])
            
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

            gm_nonzero_vox = np.sum(chimap_slice[gm_msk_slice==1] != 0)
            wm_nonzero_vox = np.sum(chimap_slice[wm_msk_slice==1] != 0)

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
        
            df = pd.DataFrame(compare_chimap_rows)

        # Now plot the selected metric (optional) for all algos
        # after calculating everithing for the slices
        # To avoid multiple plots, we can do it outside the algo loop
    if metric is not None:
        plt.figure(figsize=(10, 6))
        for algo in test_folders:
            algo_df = df[df['test_folder'] == algo]
            plt.plot(algo_df['slice_index'], algo_df[metric], label=algo, color=np.random.rand(3,), marker='o')

        plt.xlabel('Slice Index')
        plt.ylabel(metric)
        plt.title(f'{metric} across slices for different algorithms')
        plt.legend()
        plt.grid()
        plt.show()

    return df