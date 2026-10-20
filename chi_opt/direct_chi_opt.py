"""Fit lung susceptibility to averaged, already-demodulated measured profiles.

Run: python code/direct_chi_opt.py --config data/chi_027/config.json
Config paths are relative to the JSON file, rather than the working directory.
"""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

VERTEBRA_LABEL_MAP = {
    **{f"C{i}": i for i in range(1, 8)},
    **{f"T{i}": i + 7 for i in range(1, 9)},
}
DEFAULT_LEVELS = list(VERTEBRA_LABEL_MAP)[1:14]  # C2 through T7 or 2 to 15


def config_paths(config, config_path):
    values = {
        "expiration_csv": config["measured"]["expiration_csv"],
        "inspiration_csv": config["measured"]["inspiration_csv"],
        "susceptibility_map": config["simulation"]["susceptibility_map"],
        "tissue_labels": config["simulation"]["tissue_labels"],
        "cord_levels": config["simulation"]["cord_levels"],
        "output_dir": config["output_dir"],
    }
    if "fft_source" in config["simulation"]:
        values["fft_source"] = config["simulation"]["fft_source"]
    paths = {}
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain a file or directory path")
        paths[name] = (config_path.parent / Path(value).expanduser()).resolve()
    return paths


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        config = json.load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("config.schema_version must be 1")
    if not isinstance(config.get("participant_id"), str) or not config["participant_id"]:
        raise ValueError("config.participant_id must be a nonempty string")
    if config["vertebral_levels"] != DEFAULT_LEVELS:
        raise ValueError("This workflow expects vertebral_levels C2 through T7, in order")
    if config["measured"]["target"] != "average":
        raise ValueError("measured.target must be 'average'")
    if config["measured"]["units"] != "Hz" or config["simulation"]["chi_units"] != "ppm":
        raise ValueError("Measured profiles must be in Hz and susceptibility maps in ppm")
    freq = config["central_frequency_mhz"]
    if isinstance(freq, bool) or not isinstance(freq, (int, float)) or not np.isfinite(freq) or freq <= 0:
        raise ValueError("central_frequency_mhz must be finite and positive")
    padding = config["simulation"]["padding_voxels"]
    if not isinstance(padding, list) or len(padding) != 3 or any(
        isinstance(p, bool) or not isinstance(p, int) or p < 0 for p in padding
    ):
        raise ValueError("padding_voxels must contain three nonnegative integers (per side)")
    if config["simulation"]["padding_mode"] != "constant":
        raise ValueError("This workflow requires constant padding")
    padding_value = config["simulation"].setdefault("padding_value_ppm", 0.35)
    if (isinstance(padding_value, bool) or not isinstance(padding_value, (int, float))
            or not np.isfinite(padding_value)):
        raise ValueError("simulation.padding_value_ppm must be a finite number")
    if not np.isfinite(config["fit"]["trachea_chi_ppm"]):
        raise ValueError("trachea_chi_ppm must be finite")
    for name in ("lung_label", "trachea_label"):
        value = config["fit"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["fit"]["lung_label"] == config["fit"]["trachea_label"]:
        raise ValueError("Lung and trachea labels must differ")
    simulation = config["simulation"]
    if "background_chi_ppm" in simulation:
        value = simulation["background_chi_ppm"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("simulation.background_chi_ppm must be a finite number")
        label = simulation.setdefault("background_label", 0)
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise ValueError("simulation.background_label must be a nonnegative integer")
        if label in (config["fit"]["lung_label"], config["fit"]["trachea_label"]):
            raise ValueError("Background, lung and trachea labels must differ")
    if not isinstance(config["fit"]["include_initial_comparison"], bool):
        raise ValueError("include_initial_comparison must be true or false")
    if not isinstance(config.get("outputs", {}).get("save_volumes", True), bool):
        raise ValueError("outputs.save_volumes must be true or false")
    config_paths(config, path)
    return config, path


def read_measured_profile(path, settings, levels):
    frame = pd.read_csv(path)
    column = settings["value_column"]
    if column not in frame:
        raise ValueError(f"{path}: missing column {column!r}")
    if "VertLevel" in frame:
        labels = pd.to_numeric(frame["VertLevel"], errors="raise")
        if not np.isfinite(labels.to_numpy()).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError(f"{path}: VertLevel must contain finite integers")
        if labels.duplicated().any():
            raise ValueError(f"{path}: duplicate vertebral labels")
        frame = frame.assign(VertLevel=labels).set_index("VertLevel")
        wanted = [VERTEBRA_LABEL_MAP[level] for level in levels]
        if not set(wanted).issubset(frame.index):
            raise ValueError(f"{path}: missing one or more C2-T7 levels")
        values = frame.loc[wanted, column]
    else:
        # Legacy notebook CSVs contain only dmod_B0_Hz, already sorted C2-T7. Used for the ISMRM 2027 Abstract
        if settings.get("csv_row_levels") != levels or len(frame) != len(levels):
            raise ValueError(f"{path}: unlabeled CSV must have 13 rows ordered C2-T7")
        values = frame[column]
    values = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: measured values must all be finite")
    return values


def fit_lung(measured, fixed, response):
    measured, fixed, response = [np.asarray(v, dtype=float) for v in (measured, fixed, response)]
    if measured.ndim != 1 or measured.shape != fixed.shape or measured.shape != response.shape:
        raise ValueError("Measured, fixed, and response profiles must have matching vector shapes")
    if not all(np.isfinite(v).all() for v in (measured, fixed, response)):
        raise ValueError("Fit profiles must be finite")
    denominator = float(np.dot(response, response))
    if denominator <= np.finfo(float).tiny or not np.isfinite(denominator):
        raise ValueError("Lung response has zero or invalid norm; susceptibility cannot be fitted")
    chi = float(np.dot(response, measured - fixed) / denominator)
    fitted = fixed + chi * response
    residual = fitted - measured
    return chi, fitted, residual, float(np.sqrt(np.mean(residual**2)))


def load_simulation(paths, config):
    import nibabel as nib

    images = {name: nib.load(str(paths[name])) for name in
              ("susceptibility_map", "tissue_labels", "cord_levels")}
    reference = images["susceptibility_map"]
    if len(reference.shape) != 3:
        raise ValueError("Simulation volumes must be 3D")
    for name, image in images.items():
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
            raise ValueError(f"{name}: shape/affine differs from susceptibility map; align inputs first")
    chi = reference.get_fdata()
    roi = images["tissue_labels"].get_fdata()
    mask = images["cord_levels"].get_fdata()
    for name, data in (("susceptibility", chi), ("tissue labels", roi), ("cord labels", mask)):
        if not np.isfinite(data).all():
            raise ValueError(f"{name} contains nonfinite voxels")
    if not np.equal(roi, np.floor(roi)).all() or not np.equal(mask, np.floor(mask)).all():
        raise ValueError("Segmentation and cord labels must be integers")
    for key in ("lung_label", "trachea_label"):
        if not np.any(roi == config["fit"][key]):
            raise ValueError(f"Simulation segmentation does not contain {key}")
    for level in config["vertebral_levels"]:
        if not np.any(mask == VERTEBRA_LABEL_MAP[level]):
            raise ValueError(f"Cord mask has no voxels for {level}")
    if "background_chi_ppm" in config["simulation"]:
        if not np.any(roi == config["simulation"].get("background_label", 0)):
            raise ValueError("Simulation segmentation does not contain the configured background_label")
    return chi, roi, mask, np.array(reference.header.get_zooms()[:3])


def prepare_initial_map(chi_map, roi, config):
    """Assign physical background and fixed trachea in memory; preserve input files."""
    simulation = config.get("simulation", {})
    if "background_chi_ppm" in simulation:
        label = simulation.get("background_label", 0)
        value = simulation["background_chi_ppm"]
        chi_map[roi == label] = value
        print(f"Input background: label {label} -> {value:g} ppm", flush=True)
    settings = config["fit"]
    chi_map[roi == settings["trachea_label"]] = settings["trachea_chi_ppm"]
    print(f"Fixed trachea: label {settings['trachea_label']} -> "
          f"{settings['trachea_chi_ppm']:g} ppm (initial and fitted simulations)", flush=True)


@contextmanager
def atomic_output(path):
    """Replace a result only after its replacement was written successfully."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}_",
                                     suffix=path.suffix, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_results(config, paths, exp, insp, fixed, response, initial, fit,
                 postprocessing_complete=True):
    # Importing inside of the function because yeah HAHA
    import matplotlib
    matplotlib.use("Agg")  # Save plots on a remote machine without a display.
    import matplotlib.pyplot as plt

    chi, fitted, residual, rmse = fit
    measured = (exp + insp) / 2
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "VertebralLevel": config["vertebral_levels"],
        "VertLevel": [VERTEBRA_LABEL_MAP[level] for level in config["vertebral_levels"]],
        "ExpirationProfile_Hz": exp,
        "InspirationProfile_Hz": insp,
        "MeasuredProfile_Hz": measured,
        "FixedProfile_Hz": fixed,
        "LungResponse_Hz_per_ppm": response,
        "FittedProfile_Hz": fitted,
        "Residual_Hz": residual,
        "FittedLungChi_ppm": chi,
        "FitRMSE_Hz": rmse,
    })
    summary = {"participant_id": config["participant_id"], "target": "average",
               "fitted_lung_chi_ppm": chi, "fit_rmse_hz": rmse,
               "postprocessing_complete": postprocessing_complete}
    if initial is not None:
        initial_rmse = float(np.sqrt(np.mean((initial - measured)**2)))
        frame["InitialProfile_Hz"] = initial
        summary["initial_rmse_hz"] = initial_rmse
    with atomic_output(output / "average_lung_fit.csv") as temporary:
        frame.to_csv(temporary, index=False)
    for name, value in (("fit_summary.json", summary), ("config_used.json", config)):
        with atomic_output(output / name) as temporary:
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(11, 6))
    levels = config["vertebral_levels"]
    ax.plot(levels, measured, "o--", color="black", label="Measured average")
    ax.plot(levels, fitted, "s-", label=f"Fit: lung chi={chi:.4f} ppm; RMSE={rmse:.3f} Hz")
    if initial is not None:
        ax.plot(levels, initial, "s-", label=f"Initial simulation; RMSE={initial_rmse:.3f} Hz")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(xlabel="Vertebral level", ylabel="Demodulated SC field [Hz]",
           title=f"{config['participant_id']} — averaged measured profile, unconstrained lung fit")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    try:
        with atomic_output(output / "average_lung_fit.png") as temporary:
            fig.savefig(temporary, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Lung susceptibility: {chi:.4f} ppm; RMSE: {rmse:.4f} Hz", flush=True)
    print(f"Saved results to {output}", flush=True)
    if not postprocessing_complete:
        print("Fit summary, CSV and plot saved. Volume exports and initial comparison follow.", flush=True)


def backend_provenance(module, source_root):
    """Record the files actually imported, not an assumed checkout version."""
    def file_record(path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    info = {"fft_source": str(source_root),
            "compute_fieldmap": file_record(module.__file__),
            "compute_bz_signature": str(inspect.signature(module.compute_bz)),
            "fitting_script": file_record(__file__),
            "python_version": sys.version, "numpy_version": np.__version__}
    import utils.tools as tools
    info["fitting_helpers"] = file_record(tools.__file__)
    info["padding_adapter"] = "Explicit per-axis padding and crop in utils.tools; backend called with buffer=0"
    kernel = getattr(module, "dipole_kernel", None)
    kernel_file = inspect.getsourcefile(kernel) if callable(kernel) else None
    info["dipole_kernel"] = file_record(kernel_file) if kernel_file else None
    for name, command in (("git_branch", ["branch", "--show-current"]),
                          ("git_commit", ["rev-parse", "HEAD"]),
                          ("git_tracked_changes", ["status", "--porcelain", "--untracked-files=no"])):
        try:
            result = subprocess.run(["git", "-C", str(source_root), *command],
                                    capture_output=True, text=True, timeout=5)
            value = result.stdout.strip() if result.returncode == 0 else None
            info[name] = bool(value) if name == "git_tracked_changes" and value is not None else value
        except (OSError, subprocess.TimeoutExpired):
            info[name] = None
    return info


def check_backend_frequency_order(module):
    """Fail before a large run if the active z-directed solver misorders odd grids."""
    errors = {}
    for shape in ((8, 8, 8), (9, 9, 9), (9, 8, 9), (9, 9, 8)):
        signal = np.broadcast_to(np.cos(2 * np.pi * np.arange(shape[2]) / shape[2]), shape).copy()
        field = module.compute_bz(signal, image_resolution=np.ones(3), buffer=0, mode="constant")
        field -= field.mean()
        expected = (-2 / 3) * signal
        expected -= expected.mean()
        error = float(np.max(np.abs(field - expected)))
        errors[str(shape)] = error
        if not np.isfinite(error) or error > 1e-9:
            raise ValueError(f"FFT backend failed the analytical ordering check on {shape}: {error:.3e}. "
                             "Check the selected compute_fieldmap/dipole_kernel implementation.")
    return errors


def save_nifti_volume(data, reference, path, description):
    """Export float32 voxels with matching, enabled spatial forms; keep computation float64."""
    import nibabel as nib

    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1, 0)
    header["cal_min"] = header["cal_max"] = 0
    header["descrip"] = description.encode("ascii")[:79]
    image = nib.Nifti1Image(data, reference.affine, header)
    code = int(reference.header["sform_code"]) or int(reference.header["qform_code"]) or 1
    image.set_qform(reference.affine, code=code)
    image.set_sform(reference.affine, code=code)
    nib.save(image, str(path))


def save_field_pair(field_hz, reference, output, prefix, offset_hz):
    """Write raw and demodulated maps, subtracting the offset in place after saving raw."""
    save_nifti_volume(field_hz, reference, output / f"{prefix}_field_Hz.nii.gz",
                      f"{prefix} simulated field, Hz; before SC demodulation")
    field_hz -= offset_hz
    save_nifti_volume(field_hz, reference, output / f"{prefix}_field_demod_Hz.nii.gz",
                      f"{prefix} field, Hz; C2-T7 voxel-mean demodulated")


def run_simulations(config, paths, tools, chi_map, roi, shared, measured, on_fit=None):
    """Compute only the required FFTs; reuse linear responses for the fitted volume."""
    import nibabel as nib

    settings = config["fit"]
    save_volumes = config.get("outputs", {}).get("save_volumes", True)
    output = paths["output_dir"]
    reference = nib.load(str(paths["susceptibility_map"]))
    prepare_initial_map(chi_map, roi, config)
    simulation = config.get("simulation", {})
    manifest = {
        "susceptibility_units": "ppm", "field_units": "Hz", "export_dtype": "float32",
        "central_frequency_mhz": config["central_frequency_mhz"],
        "shape": list(chi_map.shape), "affine": reference.affine.tolist(),
        "demodulation": "Subtract voxel mean over cord levels C2-T7 (labels 2-14).",
        "fitted_field_method": "fixed field + fitted lung chi * unit lung field; no additional fitted FFT",
        "initial_map": "Input lung chi; fixed trachea and configured background override applied in memory.",
        "fitted_map": "Prepared initial map with lung chi fitted; trachea and background remain fixed.",
        "trachea_chi_ppm": settings["trachea_chi_ppm"],
        "physical_padding_value_ppm": shared.get("padding_value_ppm", 0.0),
        "unit_lung_response_padding_ppm": 0.0,
        "background_override": ({"label": simulation.get("background_label", 0),
                                 "chi_ppm": simulation["background_chi_ppm"]}
                                if "background_chi_ppm" in simulation else None),
    }
    if save_volumes:
        print("Saving initial susceptibility map (ppm)...", flush=True)
        save_nifti_volume(chi_map, reference, output / "initial_susceptibility_ppm.nii.gz",
                          "Initial lung chi; fixed trachea/background, ppm")

    # Disk cache avoids retaining a second complete field during the fixed-tissue FFT.
    with tempfile.TemporaryDirectory(prefix=".lung_field_cache_", dir=output) as cache_dir:
        print("Computing unit lung response...", flush=True)
        result = tools.vertebral_response(roi_label=settings["lung_label"], roi_data=roi,
                                           return_field=save_volumes, **shared)
        if save_volumes:
            response, unit_field, unit_offset = result
            unit_path = Path(cache_dir) / "unit_lung_field.npy"
            np.save(unit_path, unit_field, allow_pickle=False)
            del result, unit_field
        else:
            response = result

        fixed_map = chi_map.copy()
        fixed_map[roi == settings["lung_label"]] = 0.0
        fixed_map[roi == settings["trachea_label"]] = settings["trachea_chi_ppm"]
        print("Computing fixed tissue profile...", flush=True)
        result = tools.extract_demodulated_sc_profile(chi_map=fixed_map,
                                                       return_field=save_volumes, **shared)
        del fixed_map
        if save_volumes:
            fixed, field, fixed_offset = result
            del result
        else:
            fixed = result
        fit = fit_lung(measured, fixed, response)
        if on_fit is not None:
            # Persist the fit before full-volume reconstruction, exports or the next FFT.
            on_fit(fixed, response, fit)
        if save_volumes:
            unit_field = np.load(unit_path, mmap_mode="r", allow_pickle=False)
            try:
                for start in range(0, field.shape[0], 8):
                    slab = slice(start, start + 8)
                    field[slab] += fit[0] * unit_field[slab]
            finally:
                unit_field._mmap.close()
                del unit_field
            offset = float(np.mean(field[shared["dmod_mask"] == 1]))
            print("Saving fitted simulated fields (Hz)...", flush=True)
            save_field_pair(field, reference, output, "fitted", offset)
            manifest["fitted_demodulation_offset_hz"] = offset
            manifest["unit_lung_demodulation_offset_hz_per_ppm"] = unit_offset
            manifest["fixed_demodulation_offset_hz"] = fixed_offset
            del field

    initial = None
    if settings["include_initial_comparison"] or save_volumes:
        print("Computing initial profile with the same simulation settings...", flush=True)
        result = tools.extract_demodulated_sc_profile(chi_map=chi_map,
                                                       return_field=save_volumes, **shared)
        if save_volumes:
            initial_profile, field, offset = result
            del result
            print("Saving initial simulated fields (Hz)...", flush=True)
            save_field_pair(field, reference, output, "initial", offset)
            manifest["initial_demodulation_offset_hz"] = offset
            del field
        else:
            initial_profile = result
        if not np.isfinite(initial_profile).all():
            raise ValueError("Initial simulated profile contains nonfinite values")
        if settings["include_initial_comparison"]:
            initial = initial_profile
    if save_volumes:
        chi_map[roi == settings["lung_label"]] = fit[0]
        chi_map[roi == settings["trachea_label"]] = settings["trachea_chi_ppm"]
        print("Saving fitted susceptibility map (ppm)...", flush=True)
        save_nifti_volume(chi_map, reference, output / "fitted_susceptibility_ppm.nii.gz",
                          "Fitted susceptibility distribution, ppm")
        manifest["fitted_lung_chi_ppm"] = fit[0]
        manifest["files"] = ["initial_susceptibility_ppm.nii.gz", "fitted_susceptibility_ppm.nii.gz",
                             "initial_field_Hz.nii.gz", "initial_field_demod_Hz.nii.gz",
                             "fitted_field_Hz.nii.gz", "fitted_field_demod_Hz.nii.gz"]
        (output / "volume_outputs.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                                    encoding="utf-8")
    return fixed, response, initial, fit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Participant JSON configuration")
    parser.add_argument("--fft-source", type=Path,
                        help="FFT Git checkout containing functions/compute_fieldmap.py; overrides simulation.fft_source")
    parser.add_argument("--output-dir", type=Path, help="Override output directory for a separate run")
    parser.add_argument("--no-save-volumes", action="store_true",
                        help="Disable the six NIfTI inspection outputs (enabled by default)")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Validate config, measured CSVs and volume geometry without computing FFTs")
    parser.add_argument("--check-measured", action="store_true",
                        help="Validate config and measured CSVs; simulation files can be absent")
    args = parser.parse_args(argv)
    try:
        config, config_path = load_config(args.config)
        if args.fft_source is not None:
            config["simulation"]["fft_source"] = str(args.fft_source.expanduser().resolve())
        if args.output_dir is not None:
            config["output_dir"] = str(args.output_dir.expanduser().resolve())
        config.setdefault("outputs", {}).setdefault("save_volumes", True)
        if args.no_save_volumes:
            config["outputs"]["save_volumes"] = False
        paths = config_paths(config, config_path)
        print(f"Participant: {config['participant_id']}; target: average", flush=True)
        for note in config.get("provenance", {}).get("warnings", []):
            print(f"Configuration warning: {note}", flush=True)
        levels = config["vertebral_levels"]
        exp = read_measured_profile(paths["expiration_csv"], config["measured"], levels)
        insp = read_measured_profile(paths["inspiration_csv"], config["measured"], levels)
        measured = (exp + insp) / 2
        print(f"Measured profiles valid: {len(measured)} levels (C2-T7), Hz", flush=True)
        missing = [str(paths[name]) for name in ("susceptibility_map", "tissue_labels", "cord_levels")
                   if not paths[name].is_file()]
        if args.check_measured:
            if missing:
                print("Simulation files still needed:\n" + "\n".join(missing))
            return 0
        if missing:
            raise FileNotFoundError("Simulation files still needed:\n" + "\n".join(missing))
        import utils.tools as tools_module
        fft_source = paths.get("fft_source", Path(__file__).resolve().parent / "susceptibility-to-fieldmap-fft")
        fft_source = fft_source.resolve()
        backend = tools_module.configure_fieldmap_source(fft_source)
        config["simulation"]["fft_source"] = str(fft_source)
        print(f"FFT module: {backend.__file__}", flush=True)
        provenance = backend_provenance(backend, fft_source)
        if provenance["git_branch"]:
            print(f"FFT branch: {provenance['git_branch']}; commit: {provenance['git_commit']}", flush=True)
        if not args.check_inputs:
            provenance["frequency_order_check_max_errors"] = check_backend_frequency_order(backend)
            print("FFT even/odd frequency-order checks passed.", flush=True)
        chi_map, roi, metric_mask, resolution = load_simulation(paths, config)
        if args.check_inputs:
            print("All inputs and simulation grids are valid. No FFTs run.")
            return 0
        simulation = config["simulation"]
        padding_value = simulation["padding_value_ppm"]
        padded_shape = [int(n + 2 * p) for n, p in zip(chi_map.shape, simulation["padding_voxels"])]
        provenance["padding"] = {"voxels_each_side": simulation["padding_voxels"],
                                 "mode": simulation["padding_mode"],
                                 "constant_value_ppm": padding_value, "padded_shape": padded_shape,
                                 "unit_lung_response_constant_value_ppm": 0.0}
        print(f"FFT padding: {simulation['padding_voxels']} voxels per side, constant {padding_value:g} ppm; "
              f"padded shape: {padded_shape}", flush=True)
        dmod_mask = np.isin(metric_mask, [VERTEBRA_LABEL_MAP[level] for level in levels])
        shared = dict(image_res=resolution, central_freq=config["central_frequency_mhz"],
                      dmod_mask=dmod_mask, metric_mask=metric_mask, vertebral_levels=levels,
                      vertebra_label_map=VERTEBRA_LABEL_MAP,
                      buffer=tuple(simulation["padding_voxels"]), mode=simulation["padding_mode"],
                      padding_value_ppm=padding_value)
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        (paths["output_dir"] / "run_provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (paths["output_dir"] / "config_used.json").write_text(
            json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        def checkpoint_fit(fixed, response, fit):
            save_results(config, paths, exp, insp, fixed, response, None, fit,
                         postprocessing_complete=False)

        fixed, response, initial, fit = run_simulations(
            config, paths, tools_module, chi_map, roi, shared, measured,
            on_fit=checkpoint_fit)
        save_results(config, paths, exp, insp, fixed, response, initial, fit)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
