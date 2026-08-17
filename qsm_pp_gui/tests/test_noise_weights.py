from pathlib import Path

import nibabel as nib
import numpy as np

from qsm_pp_gui.bgfr import PIPELINES
from qsm_pp_gui.dipole import PIPELINES as DI_PIPELINES
from qsm_pp_gui.utils.noise_weights import create_noise_and_weights


def test_noise_and_weights_are_callable_and_masked(tmp_path: Path) -> None:
    magnitude = np.ones((2, 2, 2, 2), dtype=np.float64)
    magnitude[..., 1] = 2
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, :, :] = 1
    magnitude_path, mask_path = tmp_path / "mag.nii.gz", tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(magnitude, np.eye(4)), magnitude_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)

    outputs = create_noise_and_weights(magnitude_path, mask_path, [0.01, 0.02], tmp_path / "maps", "sub-01")

    assert all(path.is_file() for path in outputs.values())
    weights = nib.load(outputs["weights"]).get_fdata()
    assert np.all(weights[mask == 0] == 0)
    assert np.median(weights[mask > 0]) == 1


def test_bgfr_pipeline_choices_cover_default_and_optimized() -> None:
    assert len(PIPELINES["comp_bgfr"]) == 10
    assert all(name.startswith("def_") for name in PIPELINES["default"])
    assert all(name.startswith("opt_") for name in PIPELINES["optimized"])


def test_di_pipeline_choices_cover_all_parameter_families() -> None:
    assert len(DI_PIPELINES["comp_di"]) == 12
    assert len(DI_PIPELINES["default"]) == 5
    assert len(DI_PIPELINES["optimized"]) == 5
    assert DI_PIPELINES["automatic"] == ["auto_iLSQR", "auto_closedForm"]
