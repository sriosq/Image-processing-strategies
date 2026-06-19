from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# Initial code provided by Sam Ruttgaizer

def compare_mwf_qsm(
    qsm_path: Path | str,
    mwf_path: Path | str,
    wm_mask_path: Path | str,
    output_path: Path | str,
    mode: str = 'slicewise',  # 'slicewise' or 'voxelwise'
):
    qsm_path = Path(qsm_path)
    mwf_path = Path(mwf_path)
    wm_mask_path = Path(wm_mask_path)
    output_path = Path(output_path)

    for p in (qsm_path, mwf_path, wm_mask_path):
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    qsm = nib.load(str(qsm_path)).get_fdata()*1000  # To pass to PPB
    mwf = nib.load(str(mwf_path)).get_fdata()
    wm_prob = nib.load(str(wm_mask_path)).get_fdata()

    if mode == 'voxelwise':
        # Flatten everything, use WM prob as a boolean mask (>0 to exclude background)
        wm_voxels = (wm_prob > 0) & (mwf > 0.01)
        mwf_vals = mwf[wm_voxels]
        qsm_vals = qsm[wm_voxels]
        weights = wm_prob[wm_voxels]

        # Weighted correlation via weighted covariance
        rho, p_spearman = stats.spearmanr(mwf_vals, qsm_vals)
        r, p_pearson = stats.pearsonr(mwf_vals, qsm_vals)

        df = pd.DataFrame({'mwf': mwf_vals, 'qsm': qsm_vals, 'wm_prob': weights})

    elif mode == 'slicewise':
        rows = []
        n_slices = qsm.shape[2]
        for z in range(n_slices):
            w = wm_prob[:, :, z]
            if w.sum() == 0:
                continue  # skip slices with no WM
            mwf_mean = np.average(mwf[:, :, z], weights=w)
            qsm_mean = np.average(qsm[:, :, z], weights=w)
            rows.append({'slice': z, 'mwf_mean': mwf_mean, 'qsm_mean': qsm_mean})

        df = pd.DataFrame(rows)
        mwf_vals = df['mwf_mean']
        qsm_vals = df['qsm_mean']

        rho, p_spearman = stats.spearmanr(mwf_vals, qsm_vals)
        r, p_pearson = stats.pearsonr(mwf_vals, qsm_vals)

    else:
        raise ValueError("mode must be 'slicewise' or 'voxelwise'")

    print(f"\nSpearman ρ = {rho:.3f}, p = {p_spearman}")
    print(f"Pearson  r = {r:.3f}, p = {p_pearson}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))

    if mode == 'voxelwise':
        ax.scatter(mwf_vals, qsm_vals, c=weights, cmap='Blues',
                   s=2, alpha=0.3, rasterized=True)
        ax.set_title('MWF vs. QSM — Voxelwise (WM only)', fontsize=12, fontweight='bold')
    else:
        ax.scatter(mwf_vals, qsm_vals, color='#1F45C2', s=15, alpha=0.7)
        ax.set_title('MWF vs. QSM — Slicewise (WM only)', fontsize=12, fontweight='bold')

    m, b = np.polyfit(mwf_vals, qsm_vals, 1)
    x_line = np.linspace(mwf_vals.min() - 0.01, mwf_vals.max() + 0.01, 100)
    ax.plot(x_line, m * x_line + b, 'k--', linewidth=1, alpha=0.6)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Myelin Water Fraction', fontsize=11)
    ax.set_ylabel('QSM (ppb)', fontsize=11)

    ax.text(0.97, 0.05,
            f"Spearman ρ = {rho:.2f}, p = {p_spearman:.3f}\nPearson r = {r:.2f}, p = {p_pearson:.3f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray'))

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved to: {output_path}")

    return {
        "df": df,
        "spearman_rho": rho,
        "spearman_p": p_spearman,
        "pearson_r": r,
        "pearson_p": p_pearson,
        "output_path": output_path,
    }