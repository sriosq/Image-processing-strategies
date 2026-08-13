from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# Initial code provided by Sam Ruttgaizer

def check_p(p):
    if p < 0.00001:
        return "p << 0.01"
    else:
        return f"p = {p:.4f}"

def compare_mwf_qsm(
    qsm_path: Path | str,
    mwf_path: Path | str,
    wm_mask_path: Path | str,
    output_path: Path | str,
    stat: str | None = "pearson", # Can use pearson, spearman, "both" or None to not display any stats
    mode: str = 'slicewise',  # 'slicewise' or 'voxelwise'
    thr: int = 0, # Threshold for WM probability mask tresholding, default binarizes the mask
    y_label_var: str = "Chi-map [ppb]" # Assuming the input QSM path is a qsm image, but can be used with other maps too.
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
        wm_voxels = (wm_prob > thr) #& (mwf > 0.01)
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
        #ax.set_title('MWF vs. QSM — Voxelwise (WM only)', fontsize=12, fontweight='bold')
    else:
        ax.scatter(mwf_vals, qsm_vals, color='#1F45C2', s=15, alpha=0.7)
        ax.set_title('MWF vs. QSM — Slicewise (WM only)', fontsize=12, fontweight='bold')

    m, b = np.polyfit(mwf_vals, qsm_vals, 1)
    x_line = np.linspace(mwf_vals.min() - 0.01, mwf_vals.max() + 0.01, 100)
    ax.plot(x_line, m * x_line + b, 'k--', linewidth=1, alpha=0.6)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Myelin Water Fraction', fontsize=18)
    ax.set_ylabel(y_label_var, fontsize=18)

    if stat == "pearson":
        ax.text(0.97, 0.95,
            f"Pearson r = {r:.2f}, {check_p(p_pearson)}",
            transform=ax.transAxes, ha='right', va='top', fontsize=16,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray'))
    elif stat == "spearman":
        ax.text(0.97, 0.95,
            f"Spearman ρ = {rho:.2f}, {check_p(p_spearman)}",
            transform=ax.transAxes, ha='right', va='top', fontsize=16,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray'))
    elif stat == "both":
        ax.text(0.97, 0.95,
            f"Spearman ρ = {rho:.2f}, {check_p(p_spearman)}\nPearson r = {r:.2f}, {check_p(p_pearson)}",
            transform=ax.transAxes, ha='right', va='top', fontsize=16,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray'))
    elif stat is None or stat == "none":
        pass  # no text box — clean scatter
    else:
        raise ValueError("stat must be 'pearson', 'spearman', 'both', or None")
        
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

def spinal_permutation_test(chi_arr, mwf_arr, n_perm=10000, seed=0, alternative='two-sided'):
    rng = np.random.default_rng(seed)
    valid = ~np.isnan(chi_arr) & ~np.isnan(mwf_arr)
    r_obs, _ = stats.pearsonr(chi_arr[valid], mwf_arr[valid])

    n_levels = chi_arr.shape[0]
    null_r = np.empty(n_perm)
    for i in range(n_perm):
        lshift = rng.integers(0, n_levels)
        qshift = rng.integers(0, 4)
        mwf_shift = np.roll(mwf_arr, shift=lshift, axis=0)
        mwf_shift = np.roll(mwf_shift, shift=qshift, axis=1)
        v = ~np.isnan(chi_arr) & ~np.isnan(mwf_shift)
        null_r[i], _ = stats.pearsonr(chi_arr[v], mwf_shift[v])

    if alternative == 'two-sided':
        p = np.mean(np.abs(null_r) >= np.abs(r_obs))
    elif alternative == 'greater':
        p = np.mean(null_r >= r_obs)
    else:
        p = np.mean(null_r <= r_obs)
    return r_obs, p, null_r

def build_level_quadrant_means(chimap_data, mwf_data, anterior_wm_data, posterior_wm_data,
                                 right_wm_data, left_wm_data, levels_data):
    quad_order = ['A', 'R', 'P', 'L'] 
    quad_masks = {
        'A': anterior_wm_data.astype(bool),
        'R': right_wm_data.astype(bool),
        'P': posterior_wm_data.astype(bool),
        'L': left_wm_data.astype(bool),
    }

    # exclusivity check - do this once, don't skip it
    overlap = np.zeros(anterior_wm_data.shape, dtype=int)
    for m in quad_masks.values():
        overlap += m.astype(int)
    n_overlap = (overlap > 1).sum()
    if n_overlap > 0:
        print(f"WARNING: {n_overlap} voxels belong to >1 quadrant mask. Fix before trusting means.")

    unique_levels = sorted(int(l) for l in np.unique(levels_data) if l > 0)
    n_levels = len(unique_levels)
    chi_arr = np.full((n_levels, 4), np.nan)
    mwf_arr = np.full((n_levels, 4), np.nan)

    for li, lvl in enumerate(unique_levels):
        level_mask = levels_data == lvl
        for qi, qname in enumerate(quad_order):
            sel = quad_masks[qname] & level_mask
            if sel.sum() == 0:
                continue
            chi_arr[li, qi] = chimap_data[sel].mean()
            mwf_arr[li, qi] = mwf_data[sel].mean()

    return chi_arr, mwf_arr, unique_levels