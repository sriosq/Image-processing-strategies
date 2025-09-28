import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

    
def display_quantMap(quant_data, map_type, cut, colormap, img_class=None, slice_index=None, cmap_min=None, cmap_max=None):
    """
    Displays an axial slice of a QSM NIfTI image with a colorbar.

    Parameters:
    - qsm_data: Load the QSM map data (already in numpy array form)
    - slice_index (int, optional): Index of the slice to display. Defaults to the middle slice.
    - colormap (str): Colormap for display (default: "jet").
    """
    
        # Choose the slice index if not given
    if slice_index is None:
        if cut == 'axial':
            slice_index = quant_data.shape[2] // 2
        elif cut == 'sagittal':
            slice_index = quant_data.shape[0] // 2
        elif cut == 'coronal':
            slice_index = quant_data.shape[1] // 2

       # Extract the slice
    if cut == 'axial':
        quant_slice = quant_data[:, :, slice_index]
    elif cut == 'sagittal':
        quant_slice = quant_data[slice_index, :, :]
    elif cut == 'coronal':
        quant_slice = np.rot90(quant_data[:, slice_index, :])
    else:
        raise ValueError("Invalid cut type. Use 'axial', 'sagittal', or 'coronal'.")
    
    # --- Interactive crop selection if requested ---
    if img_class == "custom":
            
            # show image and let user select two points
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(quant_slice.T, cmap=colormap, origin="lower",
                      vmin=cmap_min, vmax=cmap_max)
            ax.set_title("Click two corners to define crop region")
            pts = plt.ginput(2)  # wait for 2 clicks
            plt.close(fig)

            (x1, y1), (x2, y2) = pts
            x_min, x_max = sorted([int(x1), int(x2)])
            y_min, y_max = sorted([int(y1), int(y2)])
            zoom_region = (x_min, x_max, y_min, y_max)

            print(f"Selected zoom region: x=({x_min},{x_max}), y=({y_min},{y_max})")

# Final display 
    plt.figure(figsize=(6, 6))
    plt.imshow(quant_slice.T, cmap=colormap, origin="lower",
               vmin=cmap_min, vmax=cmap_max)
    
    if img_class == "custom" and zoom_region:
        x_min, x_max, y_min, y_max = zoom_region
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    
    if map_type == 'pd':
        plt.title(f"PD Slice ~C6")
        plt.colorbar(orientation='horizontal')
        plt.colorbar(label=" PD (% of H20)")
    elif map_type == 't1':
        plt.title(f"T1 Slice ~C6")
        plt.colorbar(orientation='horizontal')
        plt.colorbar(label=" T1 (ms)")
    elif map_type == 't2s':
        plt.title(f"T2* Slice ~C6")
        plt.colorbar(orientation='horizontal')
        plt.colorbar(label=" T2* (ms)")
    elif map_type == 'sus':
        plt.title(f"Susceptibility map Slice ~C6")
        plt.colorbar(orientation='horizontal')
        plt.colorbar(label=r" $\chi$ (ppm)")

    plt.axis("off")
    plt.show()
    gm_mean, wm_mean = calculate_masked_mean(quant_data)
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

def select_crop_region(img_data, cut = None, slice_index=None):
    
    data = img_data

    # Choose slice
    if slice_index is None:
        if cut == 'axial':
            slice_index = data.shape[2] // 2
        elif cut == 'sagittal':
            slice_index = data.shape[0] // 2
        elif cut == 'coronal':
            slice_index = data.shape[1] // 2

    # Extract slice
    if cut == 'axial':
        slice_data = data[:, :, slice_index]
    elif cut == 'sagittal':
        slice_data = data[slice_index, :, :]
    elif cut == 'coronal':
        slice_data = np.rot90(data[:, slice_index, :])
    else:
        raise ValueError("Invalid cut type: use 'axial', 'sagittal', or 'coronal'.")

    # Show image for selection
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(slice_data.T, cmap="gray", origin="lower")
    ax.set_title("Click two opposite corners for crop region")
    
    pts = plt.ginput(2)  # user clicks
    plt.close(fig)

    # Extract coords
    (x1, y1), (x2, y2) = pts
    x_min, x_max = sorted([int(x1), int(x2)])
    y_min, y_max = sorted([int(y1), int(y2)])

    print(f"Selected zoom region: x=({x_min},{x_max}), y=({y_min},{y_max})")
    return (x_min, x_max, y_min, y_max)



