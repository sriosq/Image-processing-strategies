import nibabel as nib
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import sys
import pandas as pd
import os
from pathlib import Path
import importlib

def chi_to_fm_ppm(input_file, output_file):

    #Using a custom command for dB0 simulation

    command = [
    "compute_fieldmap",
    "-i", input_file,
    "-o", output_file,
    "-b", "50",
    "-m", "b0SimISMRM"
    ]

    # Run the command and capture output
    result = subprocess.run(command, capture_output=True, text=True)

    # We need to return the data from the fieldmap created

    
    # Check for errors
    if result.returncode != 0:
        print("Error occurred:", result.stderr)

    else:
        print("Fieldmap simulated correctly", result.stdout)
        sim_fm_img = nib.load(output_file)
        sim_fm_data = sim_fm_img.get_fdata()
        return sim_fm_data

# Keep the FFT source beside the script, on Windows and on Rosenberg.
path_to_chi_to_fm_fft = Path(__file__).resolve().parents[1] / "susceptibility-to-fieldmap-fft"
sys.path.insert(0, str(path_to_chi_to_fm_fft))
if (path_to_chi_to_fm_fft / "functions" / "compute_fieldmap.py").is_file():
    from functions import compute_fieldmap
else:
    # An explicit external checkout can be selected without a bundled fallback.
    compute_fieldmap = None


def configure_fieldmap_source(source_root):
    """Select an explicit FFT checkout; never silently import another copy."""
    global compute_fieldmap
    source_root = Path(source_root).expanduser().resolve()
    expected = source_root / "functions" / "compute_fieldmap.py"
    if not expected.is_file():
        raise FileNotFoundError(f"FFT checkout must contain functions/compute_fieldmap.py: {source_root}")
    if compute_fieldmap is not None and Path(compute_fieldmap.__file__).resolve() == expected:
        return compute_fieldmap
    previous_path = sys.path[:]
    previous_modules = {name: module for name, module in sys.modules.items()
                        if name == "functions" or name.startswith("functions.")}
    for name in previous_modules:
        del sys.modules[name]
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    try:
        selected = importlib.import_module("functions.compute_fieldmap")
        if Path(selected.__file__).resolve() != expected:
            raise ImportError(f"Requested {expected}, but imported {selected.__file__}")
    except Exception:
        for name in list(sys.modules):
            if name == "functions" or name.startswith("functions."):
                del sys.modules[name]
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path
        raise
    compute_fieldmap = selected
    return selected


def profile_from_field(field_hz, dmod_mask, metric_mask, vertebral_levels, vertebra_label_map):
    """Return the C2-T7 voxel-mean offset and per-level demodulated profile."""
    offset = float(np.mean(field_hz[dmod_mask == 1]))
    profile = np.array([
        np.mean(field_hz[metric_mask == vertebra_label_map[level]]) - offset
        for level in vertebral_levels
    ])
    return profile, offset


def compute_field_with_padding(chi_map, image_res, buffer, mode, padding_value_ppm=0.0):
    """Adapt per-axis padding to backends whose buffer argument is scalar-only.

    Own the padding and exact crop here, so every backend receives buffer=0.
    This deliberately does not select a zerofill ratio or round the FFT grid.
    padding_value_ppm is the exterior of this map, not the backend's default.
    """
    chi_map = np.asarray(chi_map)
    if chi_map.ndim != 3:
        raise ValueError("chi_map must be three dimensional")
    if (isinstance(padding_value_ppm, (bool, np.bool_))
            or not np.isscalar(padding_value_ppm) or not np.isfinite(padding_value_ppm)):
        raise ValueError("padding_value_ppm must be a finite scalar")
    padding = np.asarray(buffer)
    if padding.ndim == 0:
        padding = np.repeat(padding, 3)
    if (padding.shape != (3,) or not np.isfinite(padding).all()
            or np.any(padding < 0) or not np.equal(padding, np.floor(padding)).all()):
        raise ValueError("buffer must be one or three nonnegative integer values")
    padding = tuple(int(value) for value in padding)
    if mode == "b0SimISMRM":
        bx, by, bz = padding
        padded = np.pad(chi_map, ((bx, bx), (0, by), (bz, bz)), mode="edge")
        padded = np.pad(padded, ((0, 0), (by, 0), (0, 0)),
                        mode="constant", constant_values=0.35)
    elif any(padding):
        options = {"constant_values": padding_value_ppm} if mode == "constant" else {}
        padded = np.pad(chi_map, tuple((p, p) for p in padding), mode=mode, **options)
    else:
        padded = chi_map
    if compute_fieldmap is None:
        raise ImportError("Configure the FFT source before computing a field")
    field = compute_fieldmap.compute_bz(padded, image_resolution=image_res,
                                         buffer=0, mode="constant")
    if field.shape != padded.shape:
        raise ValueError(f"Unpadded backend returned {field.shape}, expected {padded.shape}")
    crop = tuple(slice(p, p + n) for p, n in zip(padding, chi_map.shape))
    return field[crop]

# Define this crucial function for a +1 ppm change in chi in a given ROI, returning a response vector for all vertebral levels inside the vertebral level map
def vertebral_response(
    roi_label,
    roi_data,
    image_res,
    central_freq,
    dmod_mask,
    metric_mask,
    vertebral_levels,
    vertebra_label_map,
    buffer,
    mode,
    return_field=False,
    padding_value_ppm=0.0,
):
    """Response to a +1 ppm ROI perturbation; the exterior does not change.

    Accept the physical padding setting from shared arguments, but pad this
    derivative with zero. Nonzero physical background belongs in the fixed map.
    """

    unit_chi_map = (roi_data == roi_label).astype(float)

    response_hz = (
        compute_field_with_padding(
            unit_chi_map,
            image_res=image_res,
            buffer = buffer,
            mode = mode,
            padding_value_ppm=0.0,
        )
        * central_freq
    )

    profile, offset = profile_from_field(response_hz, dmod_mask, metric_mask,
                                         vertebral_levels, vertebra_label_map)
    if return_field:
        return profile, response_hz, offset
    return profile

def extract_demodulated_sc_profile(
    chi_map,
    image_res,
    central_freq,
    dmod_mask,
    metric_mask,
    vertebral_levels,
    vertebra_label_map,
    buffer,
    mode,
    return_field=False,
    padding_value_ppm=0.0,
):
    field_hz = (
        compute_field_with_padding(
            chi_map,
            image_res=image_res,
            buffer = buffer,
            mode = mode,
            padding_value_ppm=padding_value_ppm,
        )
        * central_freq
    )

    sc_profile, offset = profile_from_field(field_hz, dmod_mask, metric_mask,
                                            vertebral_levels, vertebra_label_map)
    if return_field:
        return sc_profile, field_hz, offset
    return sc_profile
