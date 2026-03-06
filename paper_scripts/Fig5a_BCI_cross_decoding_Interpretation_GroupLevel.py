# -*- coding: utf-8 -*-
"""
===============================================================================
Title:       Group-Level Interpretation via Sensor-Space Matching (Figure 5a)
Author:      Runhao Lu
Date:        Mar 2026

Description:
    Group-level end-to-end decoding and interpretation pipeline for the BCI 
    Competition IV-2a dataset. This script implements a novel "sensor-space 
    matching" approach to generalize idiosyncratic latent reservoir dynamics 
    across independent participants.

Methodological Pipeline:
------------------------
1.  Subject-Level Feature Extraction (Loop 1): 
    For each participant, identifies the peak decoding time via training-set 
    cross-validation, concatenates the full dataset, fits the readout, and 
    extracts the reservoir states and classifier weights.

2.  Sensor-Space Projection & Sign-Alignment (Inside Phase Loop): 
    Extracts the top N most informative units per participant and projects their 
    latent activities to the common physical EEG sensor space. Polarity is aligned 
    to prevent signal cancellation during group averaging.

3.  Global Spatial Clustering: 
    Clusters the pooled spatial topographies across all participants to identify 
    consistent, population-level spatial motifs.

4.  Grand-Average Reconstruction: 
    Maps the cluster assignments back to individual reservoirs to reconstruct and 
    visualize grand-average ERPs, TFRs, PSDs (with FOOOF), and Topomaps.
===============================================================================
"""

import os

# -----------------------------------------------------------------------------
# Environment variables to reduce multi-threading interference
# -----------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import mne
import scipy.io
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Import HeteroRC modules
# -----------------------------------------------------------------------------
from heterorc import (
    HeteroRC,
    time_resolved_decoding_heterorc
)
from heterorc_interpretation import analyze_dynamics_group  

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = r"D:\RC_proj\BCI2a\Data"
SAVE_DIR = r"C:\Users\Lenovo\Desktop\RC\Results_BCI2a\Results_group_interpretation_RH"
os.makedirs(SAVE_DIR, exist_ok=True)

SUBJECTS = range(1, 10)  

SFREQ = 100
DECIM = 1 
EFFECTIVE_SFREQ = SFREQ / DECIM

N_RESERVOIR = 800
N_FOLDS = 5

rc_params = dict(
    n_res=N_RESERVOIR, input_scaling=0.5, bias_scaling=0.5, spectral_radius=0.95,
    tau_mode=0.01, tau_sigma=0.8, tau_min=0.002, tau_max=0.08,
    bidirectional=True, merge_mode="product",
)

ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SCALE_PERCENTILE = 99
RC_RANDOM_STATE = 42

FWHM_MS = 25.0
SIGMA_POINTS = (FWHM_MS / 1000.0 * EFFECTIVE_SFREQ) / 2.355

PHASES = {
    # "Cue": (0.0, 1.25),
    "Imagination": (1.25, 3.0),
}
bci2a_class_names = ["LH", "RH", "Foot", "Tongue"]

# =============================================================================
# Helper: Montage for BCI2a
# =============================================================================
def add_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    bci2a_channels = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"
    ]
    orig_names = raw.ch_names[:22]
    rename_dict = dict(zip(orig_names, bci2a_channels))
    raw.rename_channels(rename_dict)
    for ch in raw.ch_names[22:]: raw.set_channel_types({ch: "eog"})
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
    return raw

# =============================================================================
# Data loading
# =============================================================================
def load_subject_data(sub_id: int):
    print(f"\n---> Loading Subject A0{sub_id}... ", end="")

    def load_raw_simple(fname: str) -> mne.io.BaseRaw:
        fpath = os.path.join(DATA_DIR, fname)
        raw = mne.io.read_raw_gdf(fpath, preload=True, verbose=False)
        raw = add_montage(raw)
        raw.filter(0.5, 30.0, fir_design="firwin", verbose=False)
        raw.resample(SFREQ, npad="auto", verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        return raw

    # Train
    raw_t = load_raw_simple(f"A0{sub_id}T.gdf")
    events_t, _ = mne.events_from_annotations(raw_t, event_id={"769": 1, "770": 2, "771": 3, "772": 4}, verbose=False)
    epochs_t = mne.Epochs(raw_t, events_t, tmin=-0.2, tmax=3.2, proj=False, picks="eeg", baseline=(-0.2, 0), preload=True, verbose=False)
    X_train, y_train = epochs_t.get_data(copy=True) * 1e6, epochs_t.events[:, -1]

    # Labels
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, f"A0{sub_id}E.mat"))
    y_test = np.concatenate([
        mat["data"][0][i]["y"][0, 0].flatten() if "y" in mat["data"][0][i].dtype.names else mat["data"][0][i]["classlabel"][0, 0].flatten()
        for i in range(3, 9)
    ])

    # Eval
    raw_e = load_raw_simple(f"A0{sub_id}E.gdf")
    descs = raw_e.annotations.description
    if "783" in descs: events_e, _ = mne.events_from_annotations(raw_e, event_id={"783": 100}, verbose=False)
    elif "768" in descs:
        events_start, _ = mne.events_from_annotations(raw_e, event_id={"768": 100}, verbose=False)
        events_e = events_start.copy()
        events_e[:, 0] += int(2.0 * SFREQ)
    else: raise RuntimeError("Could not find start markers.")
    if len(events_e) > 288: events_e = events_e[-288:]

    epochs_e = mne.Epochs(raw_e, events_e, tmin=-0.2, tmax=3.2, proj=False, picks="eeg", baseline=(-0.2, 0), preload=True, verbose=False)
    X_test = epochs_e.get_data(copy=True) * 1e6

    if len(X_test) != len(y_test):
        m = min(len(X_test), len(y_test))
        X_test, y_test = X_test[:m], y_test[:m]

    print("Done.")
    return X_train, y_train, X_test, y_test, epochs_t.times, epochs_t.info


# =============================================================================
# Main Pipeline
# =============================================================================
if __name__ == "__main__":
    
    phase_subject_data = {phase: [] for phase in PHASES.keys()}
    global_info = None

    # -------------------------------------------------------------------------
    # 1) Loop over all subjects to collect model features and classifiers
    # -------------------------------------------------------------------------
    for sub_id in SUBJECTS:
        try:
            X_train, y_train, X_test, y_test, times, info = load_subject_data(sub_id)
        except Exception as e:
            print(f"Skipping subject {sub_id} due to error: {e}")
            continue

        if global_info is None: global_info = info

        if DECIM > 1:
            X_train, X_test, times = X_train[:, :, ::DECIM], X_test[:, :, ::DECIM], times[::DECIM]

        # --- Peak Search (Train CV) ---
        cv_curve = time_resolved_decoding_heterorc(
            X=X_train, y=y_train, times=times, n_folds=N_FOLDS, fs=EFFECTIVE_SFREQ,
            rc_params=rc_params, scale_percentile=SCALE_PERCENTILE, rc_seed_mode="fixed",
            base_rc_random_state=RC_RANDOM_STATE, metric="accuracy", smooth_decisions=True,
            smooth_sigma_points=SIGMA_POINTS, smooth_mode="nearest", verbose=False
        )

        # --- Build FULL dataset ---
        X_full = np.concatenate([X_train, X_test], axis=0)
        y_full = np.concatenate([y_train, y_test], axis=0)

        scale_val = np.percentile(np.abs(X_full), SCALE_PERCENTILE)
        X_full_s = X_full / (scale_val if scale_val > 0 else 1.0)

        esn = HeteroRC(n_in=X_full.shape[1], random_state=RC_RANDOM_STATE, fs=EFFECTIVE_SFREQ, **rc_params)
        S_full = esn.transform(X_full_s)

        # --- Phase-specific Peak extraction and Classifier fitting ---
        for phase_name, (t_min, t_max) in PHASES.items():
            mask = (times >= t_min) & (times <= t_max)
            if not np.any(mask): continue

            window_indices = np.where(mask)[0]
            peak_offset = int(np.argmax(cv_curve[window_indices]))
            peak_idx = int(window_indices[peak_offset])
            peak_time = float(times[peak_idx])

            # Fit post-hoc interpretation readout on FULL data
            clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=ALPHAS))
            clf.fit(S_full[:, :, peak_idx], y_full)

            # Pack data for this phase
            # Pack extracted features, labels, and model instances for this specific phase.
            # This list will act as the input pool for the subsequent cross-subject 
            # sensor-space matching and global clustering phase.
            phase_subject_data[phase_name].append({
                'X': X_full,
                'S': S_full,
                'y': y_full,
                'times': times,
                'target_time': peak_time,
                'classifier': clf,
                'esn': esn
            })
            print(f"  [{phase_name}] Sub A0{sub_id} - Peak found at {peak_time:.3f}s")

    # -------------------------------------------------------------------------
    # 2) Run Group Interpretation Phase by Phase
    # -------------------------------------------------------------------------
    for phase_name, sub_data_list in phase_subject_data.items():
        if not sub_data_list: continue
        
        print(f"\n" + "="*80)
        print(f"Executing Group Interpretation for Phase: {phase_name} (N={len(sub_data_list)} subjects)")
        
        results = analyze_dynamics_group(
            subject_data_list=sub_data_list,
            info=global_info,
            n_clusters=3,
            top_n=10,                      
            phase_name=f"BCI2a Group Average - {phase_name}",
            
            # ERP settings
            erp_range=(-0.2, 3.0),
            erp_baseline_mode="mean",
            erp_baseline_range=(-0.2, 0),
            
            # TFR settings
            tfr_baseline_mode="logratio",
            tfr_baseline_range=(-0.2, 0.0),
            tfr_freqs=np.arange(2, 35, 1),
            
            # FOOOF settings
            fooof_params={"max_n_peaks": 4, "peak_width_limits": [1, 8]},
            
            # Visual settings
            plot_style="poster",
            figsize=(18, 12),
            class_names=bci2a_class_names,
            cov_window_half_width=0.1,
            
            # Workspace export
            return_results=True,
            export_virtual_sources=True
        )

        fig = results["figure"]
        fname = f"BCI2a_GroupAverage_{phase_name}_Top10_3Clusters.png"
        fig.savefig(os.path.join(SAVE_DIR, fname), dpi=300, bbox_inches="tight")
        
    print("\nAll Group Interpretation tasks completed successfully!")