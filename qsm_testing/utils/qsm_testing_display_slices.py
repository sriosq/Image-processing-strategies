import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def display_qsm_slice(qsm_data, img_class=None, colormap="jet", slice_index=None, zoom_region=None, cmap_min=None, cmap_max=None):
    """
    Displays an axial slice of a QSM NIfTI image with a colorbar.

    Parameters:
    - qsm_data: Load the QSM map data (already in numpy array form)
    - slice_index (int, optional): Index of the slice to display. Defaults to the middle slice.
    - colormap (str): Colormap for display (default: "jet").
    """

    # Choose the middle slice if not specified
    if slice_index is None:
        slice_index = qsm_data.shape[2] // 2  # Assuming axial slicing

    # Extract the selected slice
    qsm_slice = qsm_data[:, :, slice_index]
    # Plot the slice
    plt.figure(figsize=(6, 6))
    plt.imshow(qsm_slice.T, cmap=colormap, origin="lower", vmin=cmap_min, vmax=cmap_max)  # Transpose to match orientation
    plt.colorbar(label=" $\\chi$ (ppm)")
    plt.title(f"QSM Slice {slice_index}")
    plt.axis("off")

    

    if img_class == "in_vivo":
        # Select this xlim and ylim
            plt.xlim(170, 210)
            plt.ylim(180, 210)

    elif img_class == "sim":
           # Select this xlim and ylim
            plt.xlim(160, 220)
            plt.ylim(180, 220)

    elif img_class == "swiss_sim":
          plt.xlim(140,180)
          plt.ylim(140,160)
    
    elif img_class == "weird_TEs":
            plt.xlim(30, 60)
            plt.ylim(125, 145)

    elif img_class == "custom":
            if zoom_region:
                x_min, x_max, y_min, y_max = zoom_region        
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
            else:
                raise ValueError("Using 'custom' requires zoom_region to be provided.")
    
    plt.show()


def display_local_field(local_field_data, img_class=None, colormap="bwr", slice_index=None, zoom_region=None, cmap_min=None, cmap_max=None):
    """
    Displays an axial slice of a QSM NIfTI image with a colorbar.

    Parameters:
    - qsm_data: Load the QSM map data (already in numpy array form)
    - slice_index (int, optional): Index of the slice to display. Defaults to the middle slice.
    - colormap (str): Colormap for display (default: "jet").
    """

    # Choose the middle slice if not specified
    if slice_index is None:
        slice_index = local_field_data.shape[2] // 2  # Assuming axial slicing

    # Extract the selected slice
    local_field_slice = local_field_data[:, :, slice_index]
    # Plot the slice
    plt.figure(figsize=(6, 6))
    plt.imshow(local_field_slice.T, cmap=colormap, origin="lower", vmin=cmap_min, vmax=cmap_max)  # Transpose to match orientation
    plt.colorbar(label="Hz")
    
    plt.axis("off")

    
    if img_class == "in_vivo":
        # Select this xlim and ylim
        if slice_index > 30 or slice_index < 40:
            plt.title(f"Local Field ~C2")
            plt.xlim(144, 174)
            plt.ylim(139, 164)

    elif img_class == "sim_ideal":
           # Select this xlim and ylim
        if slice_index > 30 or slice_index < 40:
            plt.title(f"Local Field ~C2")
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





