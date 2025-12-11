import numpy as np
import nibabel as nib

def compare_to_gt(in_map_path, gt_path, gm_mask_path, wm_mask_path):
    '''Compute RMSE between input map and ground truth within GM and WM masks.

    Parameters:
    in_map_path (str): Path to the input map NIfTI file.
    gt_path (str): Path to the ground truth NIfTI file.
    gm_mask_path (str): Path to the gray matter mask NIfTI file.
    wm_mask_path (str): Path to the white matter mask NIfTI file.

    Returns:
    float: Total RMSE (GM RMSE + WM RMSE).
    '''

    # Load the input map, ground truth, and masks
    in_map_data = nib.load(in_map_path).get_fdata()
    gt_map_data = nib.load(gt_path).get_fdata()

    gm_mask_data = nib.load(gm_mask_path).get_fdata()
    wm_mask_data = nib.load(wm_mask_path).get_fdata()

    pixelwise_difference = gt_map_data - in_map_data
    gm_diff = pixelwise_difference[gm_mask_data==1]
    wm_diff = pixelwise_difference[wm_mask_data==1]

    gm_mean_diff = np.mean(gm_diff)
    gm_std_diff = np.std(gm_diff)
    gm_rmse = np.sqrt(np.mean(gm_diff ** 2))

    wm_mean_diff = np.mean(wm_diff)
    wm_std_diff = np.std(wm_diff)
    wm_rmse = np.sqrt(np.mean(wm_diff ** 2))

    total_rmse = gm_rmse + wm_rmse
    print(f"GM Mean Diff: {gm_mean_diff}, GM Std Diff: {gm_std_diff}, GM RMSE: {gm_rmse}"
          f"\nWM Mean Diff: {wm_mean_diff}, WM Std Diff: {wm_std_diff}, WM RMSE: {wm_rmse}"
          f"\nTotal RMSE (GM + WM): {total_rmse}")
    
    return total_rmse