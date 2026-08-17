"""Noise-SD and dipole-inversion weighting maps for the QSM pipeline."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


class NoiseWeightsError(RuntimeError):
    """Raised when noise/weight inputs or calculated maps are invalid."""


def _load_mask(mask_path: Path, expected_shape: tuple[int, ...]) -> tuple[nib.Nifti1Image, np.ndarray]:
    try:
        image = nib.load(str(mask_path))
        mask = np.asarray(image.get_fdata(), dtype=np.float64)
    except (OSError, nib.filebasedimages.ImageFileError) as exc:
        raise NoiseWeightsError(f"Mask is not a readable NIfTI: {exc}") from exc
    if mask.shape != expected_shape:
        raise NoiseWeightsError(f"Mask shape {mask.shape} does not match magnitude shape {expected_shape}.")
    if not np.any(mask > 0):
        raise NoiseWeightsError("The mask contains no non-zero voxels.")
    return image, mask > 0


def create_noise_sd(magnitude_path: Path, mask_path: Path, echo_times_seconds: list[float], output_path: Path, force: bool = False) -> Path:
    """Calculate the normalized field-map SD used by SEPIA BGFR."""
    if output_path.is_file() and output_path.stat().st_size > 0 and not force:
        return output_path
    try:
        magnitude_image = nib.load(str(magnitude_path))
        magnitude = np.asarray(magnitude_image.get_fdata(), dtype=np.float64)
    except (OSError, nib.filebasedimages.ImageFileError) as exc:
        raise NoiseWeightsError(f"Magnitude is not a readable NIfTI: {exc}") from exc
    if magnitude.ndim != 4:
        raise NoiseWeightsError(f"Noise SD requires a 4D magnitude NIfTI; received shape {magnitude.shape}.")
    tes = np.asarray(echo_times_seconds, dtype=np.float64)
    if tes.size != magnitude.shape[3] or np.any(tes <= 0):
        raise NoiseWeightsError(f"Magnitude has {magnitude.shape[3]} echoes but {tes.size} positive echo times were supplied.")
    _, mask = _load_mask(mask_path, magnitude.shape[:3])
    signal = np.sqrt(np.sum(magnitude ** 2 * tes.reshape(1, 1, 1, -1) ** 2, axis=3))
    with np.errstate(divide="ignore", invalid="ignore"):
        noise_sd = 1.0 / signal
    noise_sd = np.nan_to_num(noise_sd, nan=0.0, posinf=0.0, neginf=0.0)
    samples = noise_sd[mask]
    median = np.median(samples)
    mad = np.median(np.abs(samples - median))
    cleaned = samples[np.abs(samples - median) <= 3 * 1.4826 * mad] if mad > 0 else samples
    norm = np.linalg.norm(cleaned)
    if not np.isfinite(norm) or norm <= 0:
        raise NoiseWeightsError("Noise SD normalization is zero or non-finite; check the magnitude and mask.")
    noise_sd /= norm
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = magnitude_image.header.copy()
    header.set_data_shape(magnitude.shape[:3])
    header.set_data_dtype(np.float64)
    nib.save(nib.Nifti1Image(noise_sd, magnitude_image.affine, header), str(output_path))
    return output_path


def create_di_weights(noise_sd_path: Path, mask_path: Path, output_path: Path, force: bool = False) -> Path:
    """Create normalized, clipped SEPIA-style dipole-inversion weights."""
    if output_path.is_file() and output_path.stat().st_size > 0 and not force:
        return output_path
    try:
        noise_image = nib.load(str(noise_sd_path))
        noise_sd = np.asarray(noise_image.get_fdata(), dtype=np.float64)
    except (OSError, nib.filebasedimages.ImageFileError) as exc:
        raise NoiseWeightsError(f"Noise SD is not a readable NIfTI: {exc}") from exc
    _, mask = _load_mask(mask_path, noise_sd.shape)
    weights = np.zeros_like(noise_sd)
    nonzero = noise_sd != 0
    weights[nonzero] = 1.0 / noise_sd[nonzero]
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    samples = weights[mask]
    q25, q75 = np.percentile(samples, [25, 75])
    norm = np.median(samples) + 3 * (q75 - q25)
    if not np.isfinite(norm) or norm <= 0:
        raise NoiseWeightsError("Weight normalization is zero or non-finite; check the noise SD and mask.")
    weights /= norm
    weights = weights - np.median(weights[mask]) + 1.0
    q25, q75 = np.percentile(weights[mask], [25, 75])
    threshold = np.median(weights[mask]) + 3 * (q75 - q25)
    weights = np.clip(weights, 0, threshold)
    weights[~mask] = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = noise_image.header.copy()
    header.set_data_dtype(np.float64)
    nib.save(nib.Nifti1Image(weights, noise_image.affine, header), str(output_path))
    return output_path


def create_noise_and_weights(magnitude_path: Path, mask_path: Path, echo_times_seconds: list[float], output_directory: Path, participant_id: str, force: bool = False) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    noise = output_directory / f"{participant_id}_desc-noise_sd.nii.gz"
    weights = output_directory / f"{participant_id}_desc-di_weights.nii.gz"
    create_noise_sd(magnitude_path, mask_path, echo_times_seconds, noise, force)
    create_di_weights(noise, mask_path, weights, force)
    return {"noise_sd": noise, "weights": weights}
