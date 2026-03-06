# -*- coding: utf-8 -*-
"""
Title:      BCI Competition IV 2a - Group-level CTG (Grand Average + Cluster Permutation)
Date:       Jan 2026
Description:
    Group-level analysis for cross-temporal generalisation matrices (TGM).

    This script:
      1) Loads per-subject results saved as .npz:
         - tgm_lda: (n_times, n_times)
         - tgm_rc : (n_times, n_times)
         - times  : (n_times,)
      2) Runs cluster-based permutation tests:
         - RC > LDA (matrix)
         - LDA > chance (matrix)
         - RC > chance (matrix)
         - Diagonal tests (1D):
             * RC > LDA
             * LDA > chance
             * RC > chance
      3) Plots:
         - Grand average matrices (LDA, RC, RC-LDA) with significance contours
         - Diagonal time series with SEM and significance bars

IMPORTANT:
    - Assumes all subjects share identical `times` (e.g., same epoching and DECIM).
      The script enforces this and will raise if mismatched.
    - Works for both:
        (a) within-session CV CTG outputs (train/test within T)
        (b) cross-session T->E CTG outputs (train on T, test on E)
      Just point SAVE_DIR to the correct folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import sem, t as tdist
from scipy.ndimage import label as nd_label
from mne.stats import permutation_cluster_1samp_test


# ================= Configuration =================
# Folder containing per-subject .npz outputs
SAVE_DIR = r'...' 
# Subjects expected
SUBJECTS = range(1, 10)

# BCI2a is 4-class -> chance = 0.25
CHANCE_LEVEL = 0.25

# Cluster permutation settings
N_PERMUTATIONS = 1000
ALPHA_CLUSTER = 0.05  # cluster-level p-threshold (after permutation)
ALPHA_T = 0.05        # pointwise threshold used to form clusters (one-sided)
TAIL = 1              # one-sided: H1 > 0

# Plot settings
CUE_OFFSET = 1.25
T_MIN, T_MAX = -0.2, 3.0  # crop for plotting (in seconds)

# Colormap ranges (adjust if needed)
VMIN_ACC, VMAX_ACC = 0.10, 0.40
VMIN_DIFF, VMAX_DIFF = -0.10, 0.10

# Output figure names
FIG_TGM = "Group_Generalisation.png"
FIG_DIAG = "Group_Diagonal.png"


# ================= Helpers =================
def _assert_times_equal(times_ref, times_new, sub_id):
    """Ensure all subjects share identical time axis (required for group stats)."""
    if len(times_ref) != len(times_new):
        raise ValueError(
            f"Time axis length mismatch for A0{sub_id}: {len(times_new)} vs {len(times_ref)}. "
            f"Check epoching and DECIM consistency."
        )
    max_abs = np.max(np.abs(times_new - times_ref))
    if max_abs > 1e-12:
        raise ValueError(
            f"Time axis values mismatch for A0{sub_id} (max abs diff={max_abs}). "
            f"Check epoching and DECIM consistency."
        )


def cluster_mask_1samp(X, baseline=0.0, tail=1, n_perm=1000, alpha_t=0.05, alpha_cluster=0.05, verbose=False):
    """
    Cluster permutation (1-sample) against a baseline.

    Parameters
    ----------
    X : ndarray
        Shape can be (n_sub, n_times) or (n_sub, n_times, n_times).
    baseline : float
        Null baseline. Test is performed on (X - baseline).
    tail : int
        1 for one-sided (greater), -1 for less, 0 for two-sided.
    n_perm : int
        Number of permutations.
    alpha_t : float
        Pointwise threshold used to form clusters (via t distribution).
    alpha_cluster : float
        Cluster-level p-value cutoff.
    """
    Xd = X - baseline
    n_sub = Xd.shape[0]
    df = n_sub - 1

    # One-sided threshold on t-statistics
    if tail == 1:
        t_threshold = tdist.ppf(1.0 - alpha_t, df)
    elif tail == -1:
        t_threshold = tdist.ppf(alpha_t, df)
    else:
        # two-sided: symmetric threshold
        t_threshold = tdist.ppf(1.0 - alpha_t / 2.0, df)

    t_obs, clusters, pvals, _ = permutation_cluster_1samp_test(
        Xd,
        n_permutations=n_perm,
        threshold=t_threshold,
        tail=tail,
        out_type="mask",
        verbose=verbose,
    )

    mask = np.zeros_like(t_obs, dtype=bool)
    for i_c, c in enumerate(clusters):
        if pvals[i_c] < alpha_cluster:
            mask[c] = True
    return mask


def crop_square(mat, idx_start, idx_stop):
    """Crop a (time x time) matrix to [idx_start:idx_stop+1] on both axes."""
    return mat[idx_start:idx_stop + 1, idx_start:idx_stop + 1]


def add_reference_lines(ax, tmin, tmax, cue_offset, times_crop):
    """Add cue offset lines, zero lines, and diagonal line within crop range."""
    if tmin <= cue_offset <= tmax:
        ax.axvline(cue_offset, color="k", linestyle="--", linewidth=1.5)
        ax.axhline(cue_offset, color="k", linestyle="--", linewidth=1.5)
    if tmin <= 0 <= tmax:
        ax.axvline(0, color="k", linestyle=":", linewidth=1.5)
        ax.axhline(0, color="k", linestyle=":", linewidth=1.5)
    # ax.plot([times_crop[0], times_crop[-1]], [times_crop[0], times_crop[-1]], "k--")


def plot_sig_bars(times, mask, y_pos, color, tmin, tmax, label=None, linewidth=3, alpha=0.6):
    """Plot contiguous significance segments as horizontal bars (cropped to [tmin, tmax])."""
    if not np.any(mask):
        return
    labeled, n_feats = nd_label(mask.astype(int))
    first_segment = True
    for i in range(1, n_feats + 1):
        idx = np.where(labeled == i)[0]
        t_seg = times[idx]
        if t_seg[-1] < tmin or t_seg[0] > tmax:
            continue
        t_start = max(t_seg[0], tmin)
        t_end = min(t_seg[-1], tmax)
        lbl = label if first_segment else None
        plt.hlines(y_pos, t_start, t_end, color=color, linewidth=linewidth, alpha=alpha, label=lbl)
        first_segment = False


# ================= Load Results =================
print(f"Loading results from: {SAVE_DIR}")

all_tgm_lda = []
all_tgm_rc = []
times = None
valid_subjects = []

for sub_id in SUBJECTS:
    fpath = os.path.join(SAVE_DIR, f"A0{sub_id}_results.npz")
    if not os.path.exists(fpath):
        print(f"  Warning: File for A0{sub_id} not found.")
        continue

    data = np.load(fpath)
    tgm_lda = data["tgm_lda"]
    tgm_rc = data["tgm_rc"]
    times_sub = data["times"]

    if times is None:
        times = times_sub
    else:
        _assert_times_equal(times, times_sub, sub_id)

    all_tgm_lda.append(tgm_lda)
    all_tgm_rc.append(tgm_rc)
    valid_subjects.append(sub_id)
    print(f"  Loaded A0{sub_id}")

if len(valid_subjects) < 2:
    raise RuntimeError("Need at least 2 subjects for group-level statistics.")

all_tgm_lda = np.asarray(all_tgm_lda)  # (n_sub, n_times, n_times)
all_tgm_rc = np.asarray(all_tgm_rc)    # (n_sub, n_times, n_times)

print(f"\nLoaded {len(valid_subjects)} subjects.")
print(f"Data shape: {all_tgm_lda.shape} (Subjects x Train_Time x Test_Time)")


# ================= Statistical Tests =================
print("\nRunning cluster permutation tests...")

# Full matrix tests
mask_sig_diff = cluster_mask_1samp(
    all_tgm_rc - all_tgm_lda,
    baseline=0.0,
    tail=TAIL,
    n_perm=N_PERMUTATIONS,
    alpha_t=ALPHA_T,
    alpha_cluster=ALPHA_CLUSTER,
)
mask_sig_lda = cluster_mask_1samp(
    all_tgm_lda,
    baseline=CHANCE_LEVEL,
    tail=TAIL,
    n_perm=N_PERMUTATIONS,
    alpha_t=ALPHA_T,
    alpha_cluster=ALPHA_CLUSTER,
)
mask_sig_rc = cluster_mask_1samp(
    all_tgm_rc,
    baseline=CHANCE_LEVEL,
    tail=TAIL,
    n_perm=N_PERMUTATIONS,
    alpha_t=ALPHA_T,
    alpha_cluster=ALPHA_CLUSTER,
)

print(f"  Significant points (matrix): LDA={mask_sig_lda.sum()}, RC={mask_sig_rc.sum()}, Diff={mask_sig_diff.sum()}")

# Diagonal time-series (n_sub, n_times)
diag_lda = np.diagonal(all_tgm_lda, axis1=1, axis2=2)
diag_rc = np.diagonal(all_tgm_rc, axis1=1, axis2=2)

mask_diag_diff = cluster_mask_1samp(
    diag_rc - diag_lda,
    baseline=0.0,
    tail=TAIL,
    n_perm=N_PERMUTATIONS,
    alpha_t=ALPHA_T,
    alpha_cluster=ALPHA_CLUSTER,
)
mask_diag_lda = cluster_mask_1samp(
    diag_lda,
    baseline=CHANCE_LEVEL,
    tail=TAIL,
    n_perm=N_PERMUTATIONS,
    alpha_t=ALPHA_T,
    alpha_cluster=ALPHA_CLUSTER,
)
mask_diag_rc = cluster_mask_1samp(
    diag_rc,
    baseline=CHANCE_LEVEL,
    tail=TAIL,
    n_perm=N_PERMUTATIONS,
    alpha_t=ALPHA_T,
    alpha_cluster=ALPHA_CLUSTER,
)

print(f"  Significant points (diagonal): LDA={mask_diag_lda.sum()}, RC={mask_diag_rc.sum()}, Diff={mask_diag_diff.sum()}")


# ================= Grand Average + Crop =================
ga_lda = np.mean(all_tgm_lda, axis=0)
ga_rc = np.mean(all_tgm_rc, axis=0)
ga_diff = np.mean(all_tgm_rc - all_tgm_lda, axis=0)

# Crop indices for plotting
idx_start = int(np.argmin(np.abs(times - T_MIN)))
idx_stop = int(np.argmin(np.abs(times - T_MAX)))
idx_start = int(np.clip(idx_start, 0, len(times) - 1))
idx_stop = int(np.clip(idx_stop, 0, len(times) - 1))
if idx_stop < idx_start:
    idx_start, idx_stop = idx_stop, idx_start

times_crop = times[idx_start:idx_stop + 1]
extent = [times_crop[0], times_crop[-1], times_crop[0], times_crop[-1]]

ga_lda_crop = crop_square(ga_lda, idx_start, idx_stop)
ga_rc_crop = crop_square(ga_rc, idx_start, idx_stop)
ga_diff_crop = crop_square(ga_diff, idx_start, idx_stop)

mask_sig_lda_crop = crop_square(mask_sig_lda, idx_start, idx_stop) if np.any(mask_sig_lda) else None
mask_sig_rc_crop = crop_square(mask_sig_rc, idx_start, idx_stop) if np.any(mask_sig_rc) else None
mask_sig_diff_crop = crop_square(mask_sig_diff, idx_start, idx_stop) if np.any(mask_sig_diff) else None

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import sem

# ================= Global Poster Style Settings =================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 
    'font.size': 18,            
    'axes.titlesize': 22,        
    'axes.labelsize': 20,       
    'xtick.labelsize': 18,       
    'ytick.labelsize': 18,        
    'legend.fontsize': 18,        
    'figure.titlesize': 26,      
    'lines.linewidth': 2.5,       
    'axes.linewidth': 1.5,        
    'xtick.major.width': 2.0,    
    'ytick.major.width': 2.0,
    'grid.linewidth': 1.0,
    'pdf.fonttype': 42,           
    'ps.fonttype': 42
})
# ================= Plot 1: Matrices (Poster Version) =================
print("\nPlotting grand average matrices (Poster Style)...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

contour_kwargs = {"colors": "black", "linewidths": 1.0, 
                  "extent": extent, "origin": "lower"}

# 1) LDA
im1 = axes[0].imshow(
    ga_lda_crop,
    origin="lower",
    extent=extent,
    interpolation="nearest", 
    vmin=VMIN_ACC,
    vmax=VMAX_ACC,
    cmap="RdBu_r",
)

axes[0].set_ylabel("Training Time (s)")
axes[0].set_xlabel("Testing Time (s)")
add_reference_lines(axes[0], T_MIN, T_MAX, CUE_OFFSET, times_crop)
cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cb1.ax.tick_params(labelsize=15)

if mask_sig_lda_crop is not None and np.any(mask_sig_lda_crop):
    axes[0].contour(mask_sig_lda_crop, levels=[0.5], **contour_kwargs)

# 2) RC
im2 = axes[1].imshow(
    ga_rc_crop,
    origin="lower",
    extent=extent,
    interpolation="nearest",
    vmin=VMIN_ACC,
    vmax=VMAX_ACC,
    cmap="RdBu_r",
)
# axes[1].set_title("HeteroRC")
# axes[1].set_xlabel("Testing Time (s)", fontweight='bold')
# axes[1].set_yticklabels([]) 
add_reference_lines(axes[1], T_MIN, T_MAX, CUE_OFFSET, times_crop)
cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
cb2.ax.tick_params(labelsize=15)

if mask_sig_rc_crop is not None and np.any(mask_sig_rc_crop):
    axes[1].contour(mask_sig_rc_crop, levels=[0.5], **contour_kwargs)

# 3) Difference
im3 = axes[2].imshow(
    ga_diff_crop,
    origin="lower",
    extent=extent,
    interpolation="nearest",
    vmin=VMIN_DIFF,
    vmax=VMAX_DIFF,
    cmap="PuOr_r",
)
if mask_sig_diff_crop is not None and np.any(mask_sig_diff_crop):
    axes[2].contour(mask_sig_diff_crop, levels=[0.5], **contour_kwargs)

# axes[2].set_title("Diff (RC - LDA)")
# axes[2].set_xlabel("Testing Time (s)")
add_reference_lines(axes[2], T_MIN, T_MAX, CUE_OFFSET, times_crop)

# Colorbar label
cb3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
cb3.set_label("Accuracy diff", fontsize=18)
cb3.ax.tick_params(labelsize=15)
# plt.suptitle(f"Cross-Temporal Generalization (N={len(valid_subjects)})", fontweight='bold', y=1.05)
save_path = os.path.join(SAVE_DIR, "Poster_" + FIG_TGM)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved: {save_path}")
plt.show()

# ================= Plot 2: Diagonal Time-series (Poster Version) =================
print("\nPlotting diagonal time-series (Poster Style)...")

# --- Colors---
color_lda = '#2980b9' # Darker Blue
color_rc  = '#c0392b' # Darker Red

mean_lda_d = np.mean(diag_lda, axis=0)
err_lda_d = sem(diag_lda, axis=0)
mean_rc_d = np.mean(diag_rc, axis=0)
err_rc_d = sem(diag_rc, axis=0)


plt.figure(figsize=(10, 6))

# Plot Lines 
plt.plot(times, mean_lda_d, label="LDA", linewidth=2.5, color=color_lda)
plt.fill_between(times, mean_lda_d - err_lda_d, mean_lda_d + err_lda_d, alpha=0.2, color=color_lda)

plt.plot(times, mean_rc_d, label="HeteroRC", linewidth=2.5, color=color_rc) # RC 稍微粗一点突出重点
plt.fill_between(times, mean_rc_d - err_rc_d, mean_rc_d + err_rc_d, alpha=0.2, color=color_rc)

# Reference Lines
plt.axhline(CHANCE_LEVEL, color="gray", linestyle="--", linewidth=1.5)
plt.axvline(0, color="k", linestyle=":", linewidth=1.5)
plt.axvline(CUE_OFFSET, color="gray", linestyle=":", linewidth=1.5)


y_base = 0.17 
bar_lw = 4.0 

plot_sig_bars(times, mask_diag_lda, y_base, color_lda, T_MIN, T_MAX, linewidth=bar_lw, alpha=0.5)
plot_sig_bars(times, mask_diag_rc, y_base + 0.015, color_rc, T_MIN, T_MAX, linewidth=bar_lw, alpha=0.8)

plot_sig_bars(times, mask_diag_diff, y_base + 0.030, "k", T_MIN, T_MAX, linewidth=bar_lw, alpha=1.0)

# plt.title(f"Diagonal Decoding Performance", fontweight='bold', pad=20)
plt.xlabel("Time (s)")
plt.ylabel("Decoding accuracy")


plt.legend(loc="upper right", frameon=False, fontsize=18) 

plt.ylim(0.14, 0.5) 
plt.xlim(T_MIN, T_MAX)
# plt.grid(True, linestyle="--", alpha=0.4, linewidth=1.5)
plt.tight_layout()

save_path = os.path.join(SAVE_DIR, "Poster_" + FIG_DIAG)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved: {save_path}")
plt.show()



# ================= Plot 3: Per-subject Diagonal (3x3 grid) + 50ms FWHM smoothing =================
color_lda = '#2980b9'  # Darker blue
color_rc  = '#c0392b'  # Darker red
color_link = '0.5'     # gray for paired lines
print("\nPlotting per-subject diagonal time-series (3x3) with 50ms FWHM smoothing...")

from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_rel

# Per-subject diagonals: (n_sub, n_times)
diag_lda = np.diagonal(all_tgm_lda, axis1=1, axis2=2)
diag_rc  = np.diagonal(all_tgm_rc,  axis1=1, axis2=2)

def fwhm_to_sigma_samples(fwhm_sec, times):
    """Convert FWHM in seconds to Gaussian sigma in samples, based on time axis."""
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt estimated from times: {dt}")
    sigma_sec = fwhm_sec / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_samp = sigma_sec / dt
    return sigma_samp

def smooth_gaussian_fwhm(curve, sigma_samp, mode="nearest"):
    """Gaussian smoothing along time axis using sigma in samples."""
    return gaussian_filter1d(curve, sigma=sigma_samp, axis=-1, mode=mode)

# ---- 50 ms FWHM smoothing ----
FWHM_SEC = 0.050
sigma_samp = fwhm_to_sigma_samples(FWHM_SEC, times)
print(f"  Smoothing: FWHM={FWHM_SEC*1000:.0f} ms -> sigma={sigma_samp:.2f} samples")

diag_lda_sm = np.zeros_like(diag_lda)
diag_rc_sm  = np.zeros_like(diag_rc)
for i in range(len(valid_subjects)):
    diag_lda_sm[i] = smooth_gaussian_fwhm(diag_lda[i], sigma_samp, mode="nearest")
    diag_rc_sm[i]  = smooth_gaussian_fwhm(diag_rc[i],  sigma_samp, mode="nearest")

# Grid settings
n_sub = len(valid_subjects)
n_rows, n_cols = 3, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6), sharex=True, sharey=True)
axes = axes.ravel()

# Common axis limits (match group plot)
xlim = (T_MIN, T_MAX)
ylim = (0.15, 0.60)

for i, sub_id in enumerate(valid_subjects):
    ax = axes[i]

    # plot smoothed curves
    ax.plot(times, diag_lda_sm[i],
            color=color_lda, linewidth=2.2, label="LDA")
    ax.plot(times, diag_rc_sm[i],
            color=color_rc, linewidth=2.6, label="HeteroRC")
    ax.axhline(CHANCE_LEVEL, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="k", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(CUE_OFFSET, color="gray", linestyle="-.", linewidth=1, alpha=0.8)

    ax.set_title(f"A0{sub_id}", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle=":", alpha=0.25)
    if i == 0:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=10
        )
    ax.set_title(f"Subject-0{sub_id}", fontsize=13)
    ax.tick_params(labelsize=11)

# Hide any unused panels
for j in range(n_sub, n_rows * n_cols):
    axes[j].axis("off")

# Shared labels
fig.text(0.5, 0.04, "Time (s)", ha="center")
# fig.text(0.04, 0.5, "Decoding accuracy", va="center", rotation="vertical")

plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.92])

save_path = os.path.join(SAVE_DIR, "Group_PerSubject_Diagonal_3x3_sm50ms.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved: {save_path}")
plt.show()


# ================= Plot 4: Peak decoding (cue vs imagery) + paired t-test + connected dot plot =================
print("\nComputing peak accuracies (cue vs imagery) and running paired t-tests...")

# Masks for time windows
mask_cue = (times >= 0.0) & (times <= CUE_OFFSET)          # 0 - 1.25s
mask_img = (times >  CUE_OFFSET) & (times <= T_MAX)        # >1.25s (cropped to T_MAX)

if not np.any(mask_cue):
    raise RuntimeError("No samples found in cue window. Check times and CUE_OFFSET.")
if not np.any(mask_img):
    raise RuntimeError("No samples found in imagery window. Check times, CUE_OFFSET, and T_MAX.")

# Peak accuracies per subject (use smoothed curves)
peak_lda_cue = np.max(diag_lda_sm[:, mask_cue], axis=1)
peak_rc_cue  = np.max(diag_rc_sm[:,  mask_cue], axis=1)
peak_lda_img = np.max(diag_lda_sm[:, mask_img], axis=1)
peak_rc_img  = np.max(diag_rc_sm[:,  mask_img], axis=1)

# Paired t-tests: RC vs LDA
t_cue, p_cue = ttest_rel(peak_rc_cue, peak_lda_cue)
t_img, p_img = ttest_rel(peak_rc_img, peak_lda_img)

print(f"  Cue (0–{CUE_OFFSET:.2f}s) peak:      t={t_cue:.3f}, p={p_cue:.4g}")
print(f"  Imagery (>{CUE_OFFSET:.2f}s) peak:   t={t_img:.3f}, p={p_img:.4g}")

# ---- Connected dot plots ----
fig, axes = plt.subplots(1, 2, figsize=(6, 5), sharey=True)

def connected_dotplot(ax, y_lda, y_rc, title):
    # closer x positions
    x_lda, x_rc = 0.5, 0.6

    # paired lines (gray, semi-transparent)
    for i in range(len(y_lda)):
        ax.plot([x_lda, x_rc],
                [y_lda[i], y_rc[i]],
                color=color_link,
                linewidth=2.0,
                alpha=0.2,
                zorder=1)

    # points
    ax.scatter(np.full_like(y_lda, x_lda, dtype=float),
               y_lda, s=80, color=color_lda, alpha=0.4, zorder=3, label="LDA")
    ax.scatter(np.full_like(y_rc, x_rc, dtype=float),
               y_rc,  s=80, color=color_rc,  alpha=0.4, zorder=3, label="HeteroRC")

    ax.set_xticks([x_lda, x_rc])
    ax.set_xticklabels(["LDA", "HeteroRC"], fontsize=18)
    ax.set_title(title, fontsize=20, pad=12)

   
connected_dotplot(axes[0], peak_lda_cue, peak_rc_cue,"Cue period" )
connected_dotplot(axes[1], peak_lda_img, peak_rc_img,"Imagery period" )

print(f"Cue peak (0–{CUE_OFFSET:.2f}s)\npaired t p={p_cue:.3g}")
print(f"Imagery peak (>{CUE_OFFSET:.2f}s)\npaired t p={p_img:.3g}")
axes[0].set_ylabel("Peak decoding accuracy", fontsize=20)
for ax in axes:
    ax.tick_params(labelsize=18)
axes[0].set_xlim(0.45, 0.65)
axes[1].set_xlim(0.45, 0.65)
axes[0].set_ylim(0.25, 0.65)
axes[1].legend(
    loc="upper right",
    frameon=False,
    fontsize=14
)
plt.tight_layout()

save_path = os.path.join(SAVE_DIR, "PeakAccuracy_CueVsImagery_PairedDots.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved: {save_path}")
plt.show()

