# -*- coding: utf-8 -*-
"""
===============================================================================
Title:       Group-Level Decoding Statistics & Visualization (Figure 4b/c)
Author:      Runhao Lu
Date:        Mar 2026

Description:
    Group-level statistical analysis and publication-ready visualization for 
    the Attentional Priority decoding results. This script compares HeteroRC 
    and LDA performance across participants for both 'Ping' and 'No-Ping' conditions.

Methodological Pipeline:
------------------------
1.  Data Aggregation: Loads individual decoding trajectories (.npz files).
2.  Cluster-Based Permutation Testing: Evaluates group-level significance against 
    the theoretical chance level (25%) using a one-tailed 1-sample cluster permutation test.
3.  Grand-Average Visualization (Fig 4b): Plots group mean accuracy with SEM 
    shading and solid horizontal bars denoting periods of significant decoding.
4.  Subject-Level & Peak Analysis (Fig 4c): Smooths individual trajectories 
    (50 ms FWHM), extracts peak decoding accuracies, and performs paired t-tests 
    to statistically compare HeteroRC against LDA.
===============================================================================
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mne.stats import permutation_cluster_1samp_test

# ================= 1. Environment & Configuration =================
RESULT_DIR = r"..."
chance = 0.25
sns.set_context("poster", font_scale=1.1)
sns.set_style("ticks")

res_files = glob.glob(os.path.join(RESULT_DIR, "subject_*_decoding.npz"))
n_subs = len(res_files)
print(f"Detected decoding results for {n_subs} subjects.")

# ================= 2. Load and Aggregate Data =================
data_keys = ['v_lda_p', 'v_lda_n', 'rc_p', 'rc_n'] # 'p_lda_p', 'p_lda_n', 
all_results = {key: [] for key in data_keys}
times = None

for f in res_files:
    data = np.load(f)
    if times is None: times = data['times']
    for key in data_keys:
        all_results[key].append(data[key])

# Convert to numpy arrays: shape (n_subjects, n_timepoints)
for key in data_keys:
    all_results[key] = np.array(all_results[key])


# ================= 3. Global Poster Style Settings =================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 18,              # Global base font size
    'axes.titlesize': 22,         # Subplot titles
    'axes.labelsize': 20,         # Axis labels
    'xtick.labelsize': 18,        # X-axis ticks
    'ytick.labelsize': 18,        # Y-axis ticks
    'legend.fontsize': 18,        # Legend font size
    'figure.titlesize': 26,       # Main figure title
    'lines.linewidth': 2.5,       # Default line width
    'axes.linewidth': 1.5,        # Axis spine width
    'xtick.major.width': 2.0,     # Tick line width
    'ytick.major.width': 2.0,
    'grid.linewidth': 1.0,
    'pdf.fonttype': 42,           # Ensure editable text in PDF export
    'ps.fonttype': 42
})

def plot_stat_panel(ax, ping_data, noping_data, times):
    """
    Execute cluster permutation test and plot single panel (Poster Version).
    """
    m_p = np.mean(ping_data, axis=0)
    err_p = stats.sem(ping_data, axis=0)
    m_n = np.mean(noping_data, axis=0)
    err_n = stats.sem(noping_data, axis=0)

    # --- Cluster-based Permutation Test (Ping > chance) ---
    X_stat = ping_data - chance
    X_noping = noping_data - chance
    
    t_threshold = stats.t.ppf(1 - 0.05, df=n_subs - 1)
    
    t_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
        X_stat, n_permutations=1000, threshold=t_threshold, tail=1, 
        out_type='mask', verbose=False
    )

    t_obs2, clusters2, p_values2, _ = permutation_cluster_1samp_test(
        X_noping, n_permutations=1000, threshold=t_threshold, tail=1, 
        out_type='mask', verbose=False
    )
    
    # --- ployyomh  ---
    
    # 1. Plot No-Ping baseline (Grey)
    ax.plot(times, m_n, color='gray', linestyle='--', label='No-Ping', linewidth=2.5)
    ax.fill_between(times, m_n - err_n, m_n + err_n, color='gray', alpha=0.2)
    
    # 2. Plot Ping results (Red)
    ax.plot(times, m_p, color='#E31A1C', label='Ping', linewidth=2.5)
    ax.fill_between(times, m_p - err_p, m_p + err_p, color='#E31A1C', alpha=0.2)

    # 3. Mark significant time windows (p < 0.05)
    # Dynamically place significance bars slightly below chance level
    y_sig_ping = chance - 0.018 
    y_sig_noping = chance - 0.025 

    if len(p_values) > 0:
        for i, p in enumerate(p_values):
            if p < 0.05:
                sig_times = times[clusters[i]]
                ax.plot(sig_times, [y_sig_ping]*len(sig_times), color='#E31A1C', 
                        linewidth=2.5, solid_capstyle='butt')

    if len(p_values2) > 0:
        for i, p in enumerate(p_values2):
            if p < 0.05:
                sig_times2 = times[clusters2[i]]
                ax.plot(sig_times2, [y_sig_noping]*len(sig_times2), color='gray', 
                        linewidth=2.5, solid_capstyle='butt')
                        
    ax.axhline(chance, color='black', linestyle=':', linewidth=2.0)
    ax.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # ax.set_title(title, fontweight='bold', pad=15)
    
    ax.set_ylim(chance - 0.05, chance + 0.15)
    
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # ax.set_xlabel('Time (s)')
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Decoding accuracy')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# ================= 4. Plotting =================

# 增加 figsize 以适应海报排版
fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True,sharey=True, constrained_layout=True)

# 1. Voltage LDA
plot_stat_panel(axes[0], all_results['v_lda_p'], all_results['v_lda_n'], 
                times) 

# 2. Voltage RC
plot_stat_panel(axes[1], all_results['rc_p'], all_results['rc_n'], 
                times)

axes[0].legend(frameon=False, loc='upper right')
axes[0].tick_params(labelbottom=False)
axes[-1].set_xlabel('Time (s)')
# plt.suptitle(f"Group-level Decoding Results (N={n_subs})", y=1.05, fontweight='bold')

# saving
save_img_path = os.path.join(RESULT_DIR, "..", "Group_Decoding_Poster_V2.png")
plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
plt.show()
 
# ================= 5. Per-subject Plot (HeteroRC only): Ping vs No-Ping =================
print("\nPlotting per-subject HeteroRC curves (Ping vs No-Ping)...")
# --- Re-load file list to keep subject order stable (optional) ---
res_files = glob.glob(os.path.join(RESULT_DIR, "subject_*_decoding.npz"))
res_files = sorted(res_files)  # ensure deterministic order
n_subs = len(res_files)

# Grid size (5x5 fits up to 25 subjects; 23 subjects -> 2 empty panels)
n_cols = 5
n_rows = int(np.ceil(n_subs / n_cols))

# Figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 18), sharex=True, sharey=True)
axes = np.asarray(axes).ravel()

# Consistent y-limits
ymin, ymax = (chance - 0.05, chance + 0.15)

for i, f in enumerate(res_files):
    ax = axes[i]
    d = np.load(f)

    # Extract per-subject curves
    # rc_p: Ping; rc_n: No-Ping
    rc_p = d["rc_p"]
    rc_n = d["rc_n"]
    t = d["times"]

    # Subject label from filename (e.g., subject_01_decoding.npz)
    base = os.path.basename(f)
    sub_label = base.replace("_decoding.npz", "")

    # Plot No-Ping
    ax.plot(t, rc_n, color="gray", linestyle="--", linewidth=2, alpha=0.8, label="No-Ping" if i == 0 else None)

    # Plot Ping
    ax.plot(t, rc_p, color="#E31A1C", linewidth=2.5, alpha=0.95, label="Ping" if i == 0 else None)

    # Reference lines
    ax.axhline(chance, color="black", linestyle=":", linewidth=1.5)
    ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.4)

    # Cosmetics
    ax.set_title(sub_label, fontsize=14, fontweight="bold", pad=6)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle=":", alpha=0.15)

    # Cleaner spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

# Turn off unused axes (if any)
for j in range(n_subs, n_rows * n_cols):
    axes[j].axis("off")

# Global labels
fig.text(0.5, 0.04, "Time (s)", ha="center", fontsize=18)
fig.text(0.04, 0.5, "Decoding accuracy", va="center", rotation="vertical", fontsize=18)

# One global legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=16)

fig.suptitle(f"Per-subject HeteroRC: Ping vs No-Ping (N={n_subs})", fontsize=22, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0.05, 0.06, 0.98, 0.94])

# Save
save_img_path = os.path.join(RESULT_DIR, "..", "PerSubject_HeteroRC_Ping_NoPing_5x5.png")
plt.savefig(save_img_path, dpi=300, bbox_inches="tight")

plt.show()


# ================= 6. Per-subject (Ping & No-Ping): LDA vs HeteroRC (smoothed) =================
print("\nPer-subject plots (Ping & No-Ping): LDA vs HeteroRC with 50ms FWHM smoothing...")

from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_rel

# ---- Consistent poster colors (same as previous dataset) ----
color_lda = '#2980b9'  # Darker blue
color_rc  = '#c0392b'  # Darker red
color_link = '0.5'     # gray for paired lines

# ---- 50 ms FWHM smoothing helpers ----
def fwhm_to_sigma_samples(fwhm_sec, times):
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt estimated from times: {dt}")
    sigma_sec = fwhm_sec / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return sigma_sec / dt

def smooth_gaussian(curve, sigma_samp, mode="nearest"):
    return gaussian_filter1d(curve, sigma=sigma_samp, axis=-1, mode=mode)

FWHM_SEC = 0.050
sigma_samp = fwhm_to_sigma_samples(FWHM_SEC, times)
print(f"  Smoothing: FWHM={FWHM_SEC*1000:.0f} ms -> sigma={sigma_samp:.2f} samples")

# ---- Prepare per-subject arrays from loaded results ----

# Shapes: (n_sub, n_times)
lda_p = all_results['v_lda_p']
lda_n = all_results['v_lda_n']
rc_p  = all_results['rc_p']
rc_n  = all_results['rc_n']

# Smooth (only for plotting + peak extraction)
lda_p_sm = np.array([smooth_gaussian(x, sigma_samp) for x in lda_p])
lda_n_sm = np.array([smooth_gaussian(x, sigma_samp) for x in lda_n])
rc_p_sm  = np.array([smooth_gaussian(x, sigma_samp) for x in rc_p])
rc_n_sm  = np.array([smooth_gaussian(x, sigma_samp) for x in rc_n])


res_files_sorted = sorted(glob.glob(os.path.join(RESULT_DIR, "subject_*_decoding.npz")))
sub_labels = []
for f in res_files_sorted:
    base = os.path.basename(f)
    sub_labels.append(base.replace("_decoding.npz", ""))
# If you want nice labels from filenames:
import re

sub_labels = [
    f"Subject-{int(re.search(r'subject_(\d+)', s).group(1)):02d}"
    for s in sub_labels
]

# Also reorder subject labels
# Sorting index (high -> low)
def extract_subject_id(label):
    """
    Extract integer subject id from strings like:
      'subject_13_session_1_1'
    Returns int (e.g., 13).
    """
    m = re.search(r"Subject-(\d+)", label)
    if m is None:
        raise ValueError(f"Cannot parse subject id from label: {label}")
    return int(m.group(1))

# subject IDs 
sub_ids = np.array([extract_subject_id(s) for s in sub_labels], dtype=int)
# sort index: ascending by subject id
sort_idx = np.argsort(sub_ids)

# reorder all per-subject arrays (smoothed)
lda_p_sm = lda_p_sm[sort_idx]
rc_p_sm  = rc_p_sm[sort_idx]
lda_n_sm = lda_n_sm[sort_idx]
rc_n_sm  = rc_n_sm[sort_idx]

# reorder ids / labels
sub_ids_sorted = sub_ids[sort_idx]
sub_labels = [f"Subject-{sid:02d}" for sid in sub_ids_sorted]

print("Sorted by subject ID ascending:", sub_labels[:5], "...")


def plot_per_subject_grid(times, lda_sm, rc_sm, condition_name, save_name):
    """Plot per-subject smoothed curves (LDA vs RC) in a grid, poster style."""
    n_subs_local = lda_sm.shape[0]
    n_cols = 6
    n_rows = int(np.ceil(n_subs_local / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()

    ymin, ymax = (0.15, 0.55)

    for i in range(n_subs_local):
        ax = axes[i]

        ax.plot(times, lda_sm[i], color=color_lda, linewidth=2.2, label="LDA" if i == 0 else None)
        ax.plot(times, rc_sm[i],  color=color_rc,  linewidth=2.6, label="HeteroRC" if i == 0 else None)

        ax.axhline(chance, color="black", linestyle=":", linewidth=1.8)
        ax.axvline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.4)

        ax.set_title(sub_labels[i], fontsize=18, pad=6)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle=":", alpha=0.15)

        # cleaner spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Legend ONLY in the first subject panel, upper right
        if i == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=14)

        ax.tick_params(labelsize=15)

    for j in range(n_subs_local, n_rows * n_cols):
        axes[j].axis("off")

    fig.text(0.5, 0.04, "Time (s)", ha="center", fontsize=24)
    fig.text(0.04, 0.5, "Decoding accuracy", va="center", rotation="vertical", fontsize=24)

    plt.tight_layout(rect=[0.05, 0.06, 0.98, 0.94])

    save_path = os.path.join(RESULT_DIR, "..", save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(save_path)}")
    plt.show()


# ---- Make per-subject grids for Ping and No-Ping ----
plot_per_subject_grid(times, lda_p_sm, rc_p_sm, "Ping",   "PerSubject_Ping_LDA_vs_RC_sm50ms_5x5.png")
plot_per_subject_grid(times, lda_n_sm, rc_n_sm, "No-Ping","PerSubject_NoPing_LDA_vs_RC_sm50ms_5x5.png")


# ================= 7. Peak decoding (Ping & No-Ping): paired t-test + connected dot plots =================
print("\nPeak decoding accuracy + paired t-tests (Ping & No-Ping): HeteroRC vs LDA...")

# Peak per subject (use smoothed curves)
peak_lda_ping = np.max(lda_p_sm, axis=1)
peak_rc_ping  = np.max(rc_p_sm,  axis=1)

peak_lda_nop  = np.max(lda_n_sm, axis=1)
peak_rc_nop   = np.max(rc_n_sm,  axis=1)

# Paired t-tests (RC vs LDA) for each condition
t_ping, p_ping = ttest_rel(peak_rc_ping, peak_lda_ping)
t_nop,  p_nop  = ttest_rel(peak_rc_nop,  peak_lda_nop)

print(f"  Ping peak:    t={t_ping:.3f}, p={p_ping:.4g}")
print(f"  No-Ping peak: t={t_nop:.3f},  p={p_nop:.4g}")


def connected_dotplot(ax, y_lda, y_rc, title):
    # closer x positions
    x_lda, x_rc = 0.5, 0.6

    # paired lines: gray + semi-transparent
    for i in range(len(y_lda)):
        ax.plot([x_lda, x_rc], [y_lda[i], y_rc[i]],
                color=color_link, linewidth=2.0, alpha=0.2, zorder=1)

    # points
    ax.scatter(np.full_like(y_lda, x_lda, dtype=float), y_lda,
               s=90, color=color_lda, zorder=3, alpha=0.4, label="LDA")
    ax.scatter(np.full_like(y_rc,  x_rc,  dtype=float), y_rc,
               s=90, color=color_rc,  zorder=3, alpha=0.4, label="HeteroRC")

    ax.set_xticks([x_lda, x_rc])
    ax.set_xticklabels(["LDA", "HeteroRC"], fontsize=18)
    ax.set_title(title, fontsize=20, pad=12)

    ax.tick_params(labelsize=18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# One figure, two panels (Ping / No-Ping)
fig, axes = plt.subplots(2,1, figsize=(4,8), sharey=True)

connected_dotplot(axes[0], peak_lda_ping, peak_rc_ping,
                  "Ping")
connected_dotplot(axes[1], peak_lda_nop,  peak_rc_nop,
                  "No-Ping")

axes[0].set_ylabel("Peak decoding accuracy", fontsize=20)
axes[1].set_ylabel("Peak decoding accuracy", fontsize=20)
axes[0].set_ylim(0.2, 0.65)
axes[0].set_xlim(0.45, 0.65)
axes[1].set_xlim(0.45, 0.65)
# legend only once (left panel)
axes[0].legend(loc="upper left", frameon=False, fontsize=14,ncol=2)

plt.tight_layout()

save_path = os.path.join(RESULT_DIR, "..", "PeakAccuracy_Ping_NoPing_PairedDots.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved: {os.path.abspath(save_path)}")
plt.show()
