use nifti::{InMemNiftiObject, NiftiVolume, ReaderOptions, NiftiHeader};
use ndarray::{Array3, Ix3};
use std::path::Path;

pub fn load_nifti_to_array(path: &str) -> (Array3<f32>, NiftiHeader) {
    let obj = ReaderOptions::new().read_file(path).expect("Failed to read NIfTI");

    let header = obj.header().clone();
    let vol = obj.volume();

    let dim = header.dim;
    let nx = dim[1] as usize;
    let ny = dim[2] as usize;
    let nz = dim[3] as usize;

    let mut array = Array3::<f32>::zeros((nx, ny, nz));

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                array[[x, y, z]] = vol.get_f32(&[x, y, z]).unwrap();
            }
        }
    }

    (array, header)
}

pub fn voxel_size_from_header(header: &NiftiHeader) -> (f32, f32, f32) {
    let px = header.pixdim[1];
    let py = header.pixdim[2];
    let pz = header.pixdim[3];
    (px, py, pz)
}


pub fn save_array_to_nifti(path: &str, data: &Array3<f32>, header: &NiftiHeader) {
    let mut new_header = header.clone();
    new_header.dim[0] = 3;
    new_header.dim[1] = data.shape()[0] as u16;
    new_header.dim[2] = data.shape()[1] as u16;
    new_header.dim[3] = data.shape()[2] as u16;

    let obj = InMemNiftiObject::from_header_and_volume(new_header, data.as_slice().unwrap());

    obj.save_to_file(path).expect("Failed to write NIfTI");
}
