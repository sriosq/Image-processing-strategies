use ndarray::{Array3, Axis, Zip};
use num_complex::Complex32;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Build a normalized spherical kernel for the SMV operator
pub fn smv_kernel(
    radius_mm: f32,
    voxel_size: (f32, f32, f32),
    shape: (usize, usize, usize),
) -> Array3<f32> {

    let (nx, ny, nz) = shape;
    let (vx, vy, vz) = voxel_size;

    let cx = nx as isize / 2;
    let cy = ny as isize / 2;
    let cz = nz as isize / 2;

    let mut ker = Array3::<f32>::zeros(shape);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let dx = (i as isize - cx) as f32 * vx;
                let dy = (j as isize - cy) as f32 * vy;
                let dz = (k as isize - cz) as f32 * vz;

                if dx*dx + dy*dy + dz*dz <= radius_mm * radius_mm {
                    ker[[i, j, k]] = 1.0;
                }
            }
        }
    }

    // Normalize kernel
    let sum: f32 = ker.sum();
    ker.mapv(|v| v / sum)
}

/// Perform a 3D FFT convolution: SMV(volume) = ifft( fft(volume) * fft(kernel) )
pub fn smv(
    volume: &Array3<f32>,
    radius_mm: f32,
    voxel_size: (f32, f32, f32),
) -> Array3<f32> {
    let shape = volume.raw_dim();
    let kernel = smv_kernel(radius_mm, voxel_size, (shape[0], shape[1], shape[2]));

    let vol_fft = fft3(volume);
    let ker_fft = fft3(&kernel);

    // Elementwise multiplication in Fourier domain
    let mut out_fft = vol_fft.clone();
    Zip::from(&mut out_fft)
        .and(&ker_fft)
        .for_each(|a, b| *a = *a * *b);

    // Back to real domain
    let out_complex = ifft3(&out_fft);
    out_complex.mapv(|c| c.re) // take real part
}

//
// -------- FFT IMPLEMENTATION (3D using 1D FFT along each axis) --------
//

fn fft3(input: &Array3<f32>) -> Array3<Complex32> {
    let mut data = input.mapv(|v| Complex32::new(v, 0.0));
    fft_axis(&mut data, Axis(0));
    fft_axis(&mut data, Axis(1));
    fft_axis(&mut data, Axis(2));
    data
}

fn ifft3(input: &Array3<Complex32>) -> Array3<Complex32> {
    let mut data = input.clone();
    ifft_axis(&mut data, Axis(0));
    ifft_axis(&mut data, Axis(1));
    ifft_axis(&mut data, Axis(2));
    data
}

fn fft_axis(data: &mut Array3<Complex32>, axis: Axis) {
    let len = data.len_of(axis);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(len);

    let mut buffer: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); len];

    for mut lane in data.lanes_mut(axis) {
        for i in 0..len {
            buffer[i] = lane[i];
        }
        fft.process(&mut buffer);
        for i in 0..len {
            lane[i] = buffer[i];
        }
    }
}

fn ifft_axis(data: &mut Array3<Complex32>, axis: Axis) {
    let len = data.len_of(axis);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_inverse(len);

    let mut buffer: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); len];

    for mut lane in data.lanes_mut(axis) {
        for i in 0..len {
            buffer[i] = lane[i];
        }
        fft.process(&mut buffer);
        for i in 0..len {
            lane[i] = buffer[i] / Complex32::new(len as f32, 0.0); // normalize
        }
    }
}
