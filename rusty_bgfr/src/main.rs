mod smv;
mod io;

use crate::smv::smv;
use crate::io::{load_nifti_to_array, save_array_to_nifti, voxel_size_from_header};

fn main() {
    let input_path = "E:/msc_data/educational_qsm/phantom/bgfr/sharp/B0.nii";
    let output_path = "smv_output_rust.nii.gz";
    let radius_mm = 1.0;

    // Load NIfTI into ndarray
    let (volume, header) = load_nifti_to_array(input_path);
    let voxel_size = voxel_size_from_header(&header);

    println!("Loaded volume with shape: {:?}", volume.shape());
    println!("Voxel size: {:?}", voxel_size);

    // Run SMV filter
    let out = smv(&volume, radius_mm, voxel_size);

    // Save as NIfTI
    save_array_to_nifti(output_path, &out, &header);

    println!("Saved filtered volume to {}", output_path);
}


