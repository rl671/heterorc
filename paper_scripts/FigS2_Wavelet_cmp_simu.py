# -*- coding: utf-8 -*-
"""
===============================================================================
Title:       EEG Decoding Simulation: Wavelet vs. Filter-Hilbert Baselines
Author:      Runhao Lu
Date:        Jan 2026
Description: 
    This script simulates EEG data under an induced oscillatory paradigm 
    and benchmarks the decoding performance of two standard Time-Frequency 
    feature extraction pipelines commonly used in M/EEG decoding:
    1. Morlet Wavelet Transform (calculated over surrounding frequencies, 
       averaged across the target band) + LDA
    2. Narrow-bandpass Filter + Hilbert Transform (envelope power) + LDA
===============================================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.time_frequency import tfr_array_morlet
from mne.filter import filter_data
from scipy.signal import hilbert
from scipy.stats import t, sem

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.stats import permutation_cluster_1samp_test
import warnings

warnings.filterwarnings("ignore")

from simulate_eeg import simulate_data

# ========================= 0. Path & Result Configuration =========================
RESULT_DIR =r".."
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    print(f"Created directory: {RESULT_DIR}")
else:
    print(f"Saving results to: {RESULT_DIR}")

# ========================= 1. Style Configuration =========================
def set_poster_style():
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("poster", font_scale=0.8)
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'axes.linewidth': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'xtick.major.size': 10,
        'ytick.major.size': 10
    })

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
mode = 'induced' 
freq = 10        

# Freq of interest for wavelet/hilbert
FOI_L=8
FOI_H=12

# ========================= 3. Main Loop =========================

group_scores_wavelet_lda = []
group_scores_hilbert_lda = []

#  LDA 
lda_pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
time_decod_wavelet_lda = SlidingEstimator(lda_pipe, scoring="accuracy", n_jobs=4)
time_decod_hilbert_lda = SlidingEstimator(lda_pipe, scoring="accuracy", n_jobs=4)

print(f"Executing Benchmarking: Wavelet vs. Hilbert ...")

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
    print(f"Subject {sub_idx+1}/{N_SUBJECTS}...", end=" ")
    
    # ================= 2.1) Morlet Wavelet (3-20Hz) -> 8-12Hz Power =================
    
    # Morlet Wavelet Parameters
    wavelet_freqs_calc = np.arange(5, 16)     
    n_cycles = wavelet_freqs_calc / 2.0       # MNE default heuristic

    power_wavelet = tfr_array_morlet(
        X.copy(), sfreq=SFREQ, freqs=wavelet_freqs_calc, n_cycles=n_cycles, 
        output='power', n_jobs=1, verbose=False
    )
    target_freq_mask = (wavelet_freqs_calc >= 8) & (wavelet_freqs_calc <= 12)
    X_wavelet_power = power_wavelet[:, :, target_freq_mask, :].mean(axis=2)
    
    scores_wlda = cross_val_multiscore(time_decod_wavelet_lda, X_wavelet_power, y, cv=n_fold, n_jobs=4, verbose=False)
    avg_wlda = scores_wlda.mean(axis=0)

    # ================= 2.2) Filter (8-12Hz) + Hilbert -> Power =================
    X_filt = filter_data(X.copy(), sfreq=SFREQ, l_freq=8.0, h_freq=12.0, verbose=False)
    X_hilbert_power = np.abs(hilbert(X_filt, axis=-1)) ** 2
    
    scores_hlda = cross_val_multiscore(time_decod_hilbert_lda, X_hilbert_power, y, cv=n_fold, n_jobs=4, verbose=False)
    avg_hlda = scores_hlda.mean(axis=0)

    
    # save
    group_scores_wavelet_lda.append(avg_wlda)
    group_scores_hilbert_lda.append(avg_hlda)
    
    print("Done.")
    
# Convert to Numpy Arrays
group_scores_wavelet_lda = np.array(group_scores_wavelet_lda)
group_scores_hilbert_lda = np.array(group_scores_hilbert_lda)

# ========================= 4. Statistics & Plotting =========================

def run_cluster_1d_greater(data, chance, n_permutations=1000, alpha_cluster=0.05, seed=0):
    data = np.asarray(data)
    n_sub = data.shape[0]
    diff = data - float(chance)
    t_thresh = t.ppf(1 - alpha_cluster, df=n_sub - 1)

    T_obs, clusters, pvals, _ = permutation_cluster_1samp_test(
        diff,
        n_permutations=n_permutations,
        threshold=t_thresh,
        tail=1,                 
        out_type="indices",     
        seed=seed,
        n_jobs=2,
    )
    return clusters, pvals

def plot_full_statistical_results(times, data_wlda, data_hlda,  save_dir=RESULT_DIR):
    set_poster_style()
    chance_level = 1.0 / N_CLASSES
    
    # Statistics
    clusters_wlda, p_wlda = run_cluster_1d_greater(data_wlda, chance_level, n_permutations=1000, alpha_cluster=0.05, seed=0)
    clusters_hlda, p_hlda = run_cluster_1d_greater(data_hlda, chance_level, n_permutations=1000, alpha_cluster=0.05, seed=0)
    
    alpha_tail = 0.05
    sig_wlda = [clu[0] for clu, pv in zip(clusters_wlda, p_wlda) if pv < alpha_tail]
    sig_hlda = [clu[0] for clu, pv in zip(clusters_hlda, p_hlda) if pv < alpha_tail]

    # Calculate Mean and SEM
    mean_wlda, sem_wlda = np.mean(data_wlda, axis=0), sem(data_wlda, axis=0)
    mean_hlda, sem_hlda = np.mean(data_hlda, axis=0), sem(data_hlda, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.axvline(0, color='k', linestyle=':', linewidth=2, alpha=0.5)
    ax.axhline(chance_level, color='gray', linestyle='--', linewidth=2, zorder=0)

    # --- Colors ---
    color_wlda = '#f39c12' # Orange/Gold (Wavelet)
    color_hlda = '#27ae60' # Green (Hilbert)
    
    # 1. Wavelet-LDA
    ax.plot(times, mean_wlda, 
            # label='Wavelet + LDA', 
            color=color_wlda, linewidth=3)
    ax.fill_between(times, mean_wlda - sem_wlda, mean_wlda + sem_wlda, color=color_wlda, alpha=0.2, lw=0)

    # 2. Hilbert-LDA
    ax.plot(times, mean_hlda, 
            # label='Filter-Hilbert + LDA', 
            color=color_hlda, linewidth=3)
    ax.fill_between(times, mean_hlda - sem_hlda, mean_hlda + sem_hlda, color=color_hlda, alpha=0.2, lw=0)

    # --- Significance Bars ---
    bar_y_start = 0.40
    step = 0.015
    
    # Wavelet-LDA Bar
    if len(sig_wlda) > 0:
        for idxs in sig_wlda:
            ax.hlines(y=bar_y_start, xmin=times[idxs[0]], xmax=times[idxs[-1]], 
                      color=color_wlda, linewidth=4, alpha=0.8)
    
    # Hilbert-LDA Bar
    if len(sig_hlda) > 0:
        for idxs in sig_hlda:
            ax.hlines(y=bar_y_start + step, xmin=times[idxs[0]], xmax=times[idxs[-1]], 
                      color=color_hlda, linewidth=4, alpha=0.8)

    # # --- Layout & Labels ---
    ax.set_xlabel('Time (s)', fontsize=28, labelpad=12)
        
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.set_ylim([0.35, 1.0])
    ax.set_xlim([times[0], times[-1]])
    
  
    ax.set_ylabel('Decoding Accuracy', fontsize=28, labelpad=12)
    ax.legend(fontsize=20, frameon=False, loc='upper right', ncol=1)
        
    # Save Figure
    save_path = os.path.join(save_dir, f'Tri_Comparison_N{N_SUBJECTS}_{mode}_freq{freq}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()

# Execute Plotting
plot_full_statistical_results(times, group_scores_wavelet_lda, group_scores_hilbert_lda)