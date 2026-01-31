# -*- coding: utf-8 -*-
"""
===============================================================================
Title:       EEG Decoding Simulation: HeteroRC vs. Linear Baselines (LDA/SVM)
Author:      Runhao Lu
Date:        Jan 2026
Description: 
    This script simulates EEG data under various physiological paradigms 
    (ERP, Induced Power, Inter-Site Phase Clustering, 1/f Slope, 1/f Intercept) and benchmarks 
    the decoding performance of Reservoir Computing (HeteroRC) against 
    classic linear models (LDA and Linear SVM).

    Key Features:
    1. Simulation of ERP, periodic (oscillatory power and ISPC) and aperiodic (1/f slope and intercept) activity.
    2. Spatially constrained modulation (e.g., posterior-only changes) within ~200-600ms time window (with jitters).
    3. Rigorous cross-validation with strictly isolated feature scaling.
    4. Cluster-based permutation testing for statistical significance.
    
===============================================================================
"""

import os
# Force single-core execution to ensure fair comparison of computation time (if needed)
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage import label
import warnings
# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

# Key functions
from heterorc import HeteroRC,time_resolved_decoding_heterorc
from simulate_eeg import simulate_data

# ========================= 0. Path & Result Configuration =========================
RESULT_DIR = r"X:\...\...\..."
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    print(f"Created directory: {RESULT_DIR}")
else:
    print(f"Saving results to: {RESULT_DIR}")

# ========================= 1. Style Configuration (Publication Ready) =========================
plt.style.use('seaborn-v0_8-paper')
sns.set_context("poster", font_scale=0.8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ========================= 2. Core Configuration =========================
N_SUBJECTS = 30
SFREQ = 100          
TMIN, TMAX = 0.0, 0.8
N_CHANNELS = 32 
N_CLASSES = 2
N_TRIALS_PER_CLASS = 40
N_TOTAL_TRIALS = N_TRIALS_PER_CLASS * N_CLASSES
noise_scale = 1.0
n_fold = 5

# --- Simulation Mode Selection ---
# Options: 'erp', 'induced', 'ispc', 'slope', 'intercept'
mode = 'erp' 
freq = 10 # for induced and ispc modulation

# RC Core Parameters
rc_params = dict(
    n_res=350,
    input_scaling=0.5,
    bias_scaling=0.5,
    spectral_radius=0.95,
    tau_mode=0.01,
    tau_sigma=0.8,
    tau_min=0.002,
    tau_max=0.08,
    bidirectional=False,
    merge_mode="average",
)
    
# ========================= 5. Main Loop with Timer =========================

group_scores_lda = []
group_scores_svm = []
group_scores_rc_smooth = []

# --- Define MNE decoders once (they will be cloned internally) ---
lda_pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
svm_pipe = make_pipeline(StandardScaler(), LinearSVC())  # linear SVM, faster than SVC(kernel='linear')

time_decod_lda = SlidingEstimator(lda_pipe, scoring="accuracy", n_jobs=4)
time_decod_svm = SlidingEstimator(svm_pipe, scoring="accuracy", n_jobs=4)

print(f"Executing Benchmarking: LDA vs. SVM vs. RC ({mode} mode)...")

for sub_idx in range(N_SUBJECTS):
    # 1) Simulate Data
    X, y, times = simulate_data(sub_idx, mode, target_freq=freq,
                                noise_scale_=noise_scale,
                                sfreq=SFREQ,
                                tmin=TMIN,
                                tmax=TMAX,
                                n_trials=N_TOTAL_TRIALS,
                                n_channels=N_CHANNELS,
                                n_classes=N_CLASSES)
    n_times = len(times)

    print(f"Subject {sub_idx+1}/{N_SUBJECTS}...", end="")
    # 2) LDA (MNE handles CV + time-resolved fitting)
    scores_lda = cross_val_multiscore(time_decod_lda, X, y, cv=n_fold, n_jobs=4,verbose=False)
    avg_lda = scores_lda.mean(axis=0)  # (n_times,)

    # 3) SVM (MNE)
    scores_svm = cross_val_multiscore(time_decod_svm, X, y, cv=n_fold, n_jobs=4,verbose=False)
    avg_svm = scores_svm.mean(axis=0)
    # 4) RC (your function; includes internal CV and optional smoothing)
    # Smoothing parameters for RC readout
    FWHM_MS = 25.0 
    sigma_points = (FWHM_MS / 2.355 / 1000.0) * SFREQ
    avg_rc_smooth = time_resolved_decoding_heterorc(
        X, y, times,
        n_folds=n_fold,
        fs=SFREQ,
        rc_params=rc_params,
        rc_seed_mode='fixed',
        smooth_decisions=True,
        smooth_sigma_points=sigma_points,
        cv_random_state=100,
        base_rc_random_state=42,  # fold seed base
        verbose=False,
    )
    
    group_scores_lda.append(avg_lda)
    group_scores_svm.append(avg_svm)
    group_scores_rc_smooth.append(avg_rc_smooth)
    
    print("Done.")
    
# Convert to Numpy Arrays
group_scores_lda = np.array(group_scores_lda)
group_scores_svm = np.array(group_scores_svm)
group_scores_rc_smooth = np.array(group_scores_rc_smooth)


# ========================= 7. Statistics & Plotting =========================
from scipy.stats import t,sem
from mne.stats import permutation_cluster_1samp_test


def run_cluster_1d_greater(data, chance, n_permutations=1000, alpha_cluster=0.05, seed=0):
    """
    data: (n_subjects, n_times)
    tests H1: mean(data - chance) > 0 with a one-tailed cluster permutation test.
    Returns significant clusters as time-index arrays plus p-values.
    """
    data = np.asarray(data)
    n_sub = data.shape[0]

    diff = data - float(chance)

    # cluster-forming threshold as t-value (one-tailed)
    t_thresh = t.ppf(1 - alpha_cluster, df=n_sub - 1)

    T_obs, clusters, pvals, _ = permutation_cluster_1samp_test(
        diff,
        n_permutations=n_permutations,
        threshold=t_thresh,
        tail=1,                 # 1 = greater-than
        out_type="indices",     # clusters are (idx_array,) tuples
        seed=seed,
        n_jobs=2,
    )
    return clusters, pvals

def plot_full_statistical_results(times, data_lda, data_svm, data_rc, save_dir=RESULT_DIR):
    set_poster_style()
    chance_level = 1.0 / N_CLASSES
    clusters_rc, p_rc = run_cluster_1d_greater(data_rc, chance_level, n_permutations=1000, alpha_cluster=0.05, seed=0)
    clusters_lda, p_lda = run_cluster_1d_greater(data_lda, chance_level, n_permutations=1000, alpha_cluster=0.05, seed=0)
    clusters_svm, p_svm = run_cluster_1d_greater(data_svm, chance_level, n_permutations=1000, alpha_cluster=0.05, seed=0)
    
    alpha_tail = 0.05
    sig_rc  = [clu[0] for clu, pv in zip(clusters_rc,  p_rc)  if pv < alpha_tail]
    sig_lda = [clu[0] for clu, pv in zip(clusters_lda, p_lda) if pv < alpha_tail]
    sig_svm = [clu[0] for clu, pv in zip(clusters_svm, p_svm) if pv < alpha_tail]

    # Calculate Mean and SEM
    mean_lda, sem_lda = np.mean(data_lda, axis=0), sem(data_lda, axis=0)
    mean_svm, sem_svm = np.mean(data_svm, axis=0), sem(data_svm, axis=0)
    mean_rc, sem_rc = np.mean(data_rc, axis=0), sem(data_rc, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    ax.axvline(0, color='k', linestyle=':', linewidth=2, alpha=0.5)
    ax.axhline(chance_level, color='gray', linestyle='--', linewidth=2, zorder=0)

    # --- Colors ---
    color_lda = '#3498db' # Blue
    color_svm = '#9b59b6' # Purple
    color_rc  = '#e74c3c' # Red
    
    # 1. LDA
    ax.plot(times, mean_lda, label='LDA', color=color_lda, linewidth=3)
    ax.fill_between(times, mean_lda - sem_lda, mean_lda + sem_lda, color=color_lda, alpha=0.2, lw=0)
    
    # 2. SVM
    ax.plot(times, mean_svm, label='SVM', color=color_svm, linewidth=3)
    ax.fill_between(times, mean_svm - sem_svm, mean_svm + sem_svm, color=color_svm, alpha=0.2, lw=0)

    # 3. RC
    ax.plot(times, mean_rc, label='HeteroRC', color=color_rc, linewidth=3)
    ax.fill_between(times, mean_rc - sem_rc, mean_rc + sem_rc, color=color_rc, alpha=0.2, lw=0)

    # --- Significance Bars (Offset for visibility) ---
    bar_y_start = 0.41
    step = 0.012
    
    # LDA Bar
    if len(sig_lda) > 0:
        for idxs in sig_lda:
            ax.hlines(y=bar_y_start, xmin=times[idxs[0]], xmax=times[idxs[-1]], 
                      color=color_lda, linewidth=4, alpha=0.8)
    # SVM Bar
    if len(sig_svm) > 0:
        for idxs in sig_svm:
            ax.hlines(y=bar_y_start + step, xmin=times[idxs[0]], xmax=times[idxs[-1]], 
                      color=color_svm, linewidth=4, alpha=0.8)
    # RC Bar
    if len(sig_rc) > 0:
        for idxs in sig_rc:
            ax.hlines(y=bar_y_start + 2*step, xmin=times[idxs[0]], xmax=times[idxs[-1]], 
                      color=color_rc, linewidth=4)

    # --- Layout & Labels ---
    if mode == 'intercept':
        ax.set_xlabel('Time (s)', fontsize=28,  labelpad=12)
    # ax.set_title(f'HeteroRC vs. Linear Baselines ({mode} mode)', fontsize=28, fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=28)
    # Fixed Y-axis Limits
    ax.set_ylim([0.4,0.9])
    ax.set_xlim([times[0], times[-1]])
    if mode == 'erp':
        ax.set_ylabel('Decoding Accuracy', fontsize=28,  labelpad=12)
        ax.legend(fontsize=22, frameon=False, loc='upper right', ncol=1)
    # Save Figure
    save_path = os.path.join(save_dir, f'Uni_Decoding_Comparison_N{N_SUBJECTS}_{mode}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()

def set_poster_style():
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'axes.linewidth': 2.5,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'xtick.major.size': 10,
        'ytick.major.size': 10
    })

# Execute
plot_full_statistical_results(times, group_scores_lda, group_scores_svm, group_scores_rc_smooth)