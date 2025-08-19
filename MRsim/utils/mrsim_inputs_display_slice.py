import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
    
def display_wb_quantMap(quant_data, map_type, cut, colormap, img_class=None, slice_index=None, zoom_region=None, cmap_min=None, cmap_max=None):
    """
    Displays an axial slice of a QSM NIfTI image with a colorbar.

    Parameters:
    - qsm_data: Load the QSM map data (already in numpy array form)
    - slice_index (int, optional): Index of the slice to display. Defaults to the middle slice.
    - colormap (str): Colormap for display (default: "jet").
    """

    # Choose the middle slice if not specified
    if slice_index is None:
        slice_index = quant_data.shape[2] // 2  # Assuming axial slicing

    # Extract the selected slice
    if cut == 'axial':
        quant_slice = quant_data[:, :, slice_index]
        
    elif cut == 'sagittal':
        quant_slice = quant_data[slice_index, :, :]
    elif cut == 'coronal':
        quant_slice = np.rot90(quant_data[:, slice_index, :])
    else:
        raise ValueError("Invalid cut type. Use 'axial', 'sagittal', or 'coronal'.")
    

    # Plot the slice
    plt.figure(figsize=(6, 6))
    plt.imshow(quant_slice.T, cmap=colormap, origin="lower", vmin=cmap_min, vmax=cmap_max)  # Transpose to match orientation
    
    if map_type == 'pd':
        plt.title(f"PD Slice ~C6")
        plt.colorbar(label=" PD (% of H20)")
    elif map_type == 't1':
        plt.title(f"T1 Slice ~C6")
        plt.colorbar(label=" T1 (ms)")
    elif map_type == 't2s':
        plt.title(f"T2* Slice ~C6")
        plt.colorbar(label=" T2* (ms)")
    elif map_type == 'sus':
        plt.title(f"Susceptibility map Slice ~C6")
        plt.colorbar(label=" $\chi$ (ppm)")

    plt.axis("off")

    if img_class == "sim_ideal":
            if cut == 'axial':
                plt.xlim(144, 174)
                plt.ylim(139, 164)
            if cut == 'sagittal':
                plt.xlim(40, 360)
                plt.ylim(150, 760)

    elif img_class == "swiss_sim":
          plt.xlim(140,180)
          plt.ylim(140,160)
    
    elif img_class == "custom":
            if zoom_region:
                x_min, x_max, y_min, y_max = zoom_region        
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
            else:
                raise ValueError("Using 'custom' requires zoom_region to be provided.")
    
    plt.show()
    gm_mean, wm_mean = calculate_masked_mean(quant_data)
    print(f"GM Mean: {gm_mean:.4f} ppm")
    print(f"WM Mean: {wm_mean:.4f} ppm")


def display_T1_slice(T1_map_data, img_class=None, colormap="gray", slice_index=None, zoom_region=None, cmap_min=None, cmap_max=None):
    """
    Displays an axial slice of a QSM NIfTI image with a colorbar.

    Parameters:
    - qsm_data: Load the QSM map data (already in numpy array form)
    - slice_index (int, optional): Index of the slice to display. Defaults to the middle slice.
    - colormap (str): Colormap for display (default: "jet").
    """

    # Choose the middle slice if not specified
    if slice_index is None:
        slice_index = T1_map_data.shape[2] // 2  # Assuming axial slicing

    # Extract the selected slice
    T1_slice = T1_map_data[:, :, slice_index]
    # Plot the slice
    plt.figure(figsize=(6, 6))
    plt.imshow(T1_slice.T, cmap=colormap, origin="lower", vmin=cmap_min, vmax=cmap_max)  # Transpose to match orientation
    plt.colorbar(label=" PD (% of H20)")
    plt.title(f"PD Slice ~C2")
    plt.axis("off")

    if img_class == "sim_ideal":
            plt.xlim(144, 174)
            plt.ylim(139, 164)

    elif img_class == "swiss_sim":
          plt.xlim(140,180)
          plt.ylim(140,160)
    
    elif img_class == "custom":
            if zoom_region:
                x_min, x_max, y_min, y_max = zoom_region        
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
            else:
                raise ValueError("Using 'custom' requires zoom_region to be provided.")
    
    plt.show()
    gm_mean, wm_mean = calculate_masked_mean(T1_map_data)
    print(f"GM Mean: {gm_mean:.4f} ppm")
    print(f"WM Mean: {wm_mean:.4f} ppm")


def load_gm_wm_masks(gm_mask_path, wm_mask_path):
    global wm_mask_data, gm_mask_data
    wm_mask_data = nib.load(wm_mask_path).get_fdata()
    gm_mask_data = nib.load(gm_mask_path).get_fdata()

def calculate_masked_mean(img_data):
    """
    Calculate the mean of the data within the GM and WM region.
    """
    tmp_gm_mean = np.mean(img_data[gm_mask_data == 1])
    tmp_wm_mean = np.mean(img_data[wm_mask_data == 1])
    return tmp_gm_mean, tmp_wm_mean




