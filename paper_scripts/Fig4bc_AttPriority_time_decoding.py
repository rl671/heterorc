# -*- coding: utf-8 -*-
"""
===============================================================================
Title:       Individual-Level Time-Resolved Decoding (Figure 4)
Author:      Runhao Lu
Date:        Mar 2026

Description:
    Time-resolved EEG decoding pipeline for the Attentional Priority dataset 
    (Duncan et al., 2023). This script evaluates the decoding performance of 
    the HeteroRC framework against a standard Linear Discriminant Analysis (LDA) 
    baseline for a 4-class spatial attention mapping task. Analyses are conducted 
    separately for 'Ping' (impulse stimulus) and 'No-Ping' (baseline) trials.

Methodological Pipeline:
------------------------
1.  Data Preprocessing: Loads raw BDF files, applies bandpass filtering, and 
    epochs data into 'Ping' and 'No-Ping' trials. Applies a baseline correction
    relative to the pre-stimulus interval.
2.  Linear Baseline (LDA): A standard sliding-window estimator evaluates 
    instantaneous spatial voltage patterns using cross-validation.
3.  HeteroRC Decoding: Projects EEG data into the heterogeneous reservoir space 
    and evaluates performance using a Ridge classifier.
4.  Temporal Smoothing: Applies a 25 ms FWHM Gaussian smoothing to decision 
    scores to mimic downstream temporal integration and enhance SNR.
===============================================================================
"""


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import SlidingEstimator, cross_val_multiscore
from joblib import Parallel, delayed
# ====== import OUR functions ======
# Make sure heterorc.py is available (same folder or installed)
from heterorc import time_resolved_decoding_heterorc

# ================= 1. Config =================
BASE_DIR = r"C:\Users\Lenovo\Desktop\RC"
DATA_DIR = r"D:\RC_proj\Duncan2023\Data"
subject_files = glob.glob(os.path.join(DATA_DIR, "subject_*.bdf"))

RESULT_DIR = os.path.join(BASE_DIR, "Results_AttentionPri_acc_baselined")
os.makedirs(RESULT_DIR, exist_ok=True)

# Event ID 
event_id_ping   = {'Loc_0': 100, 'Loc_1': 102, 'Loc_2': 104, 'Loc_3': 106}
event_id_noping = {'Loc_0': 200, 'Loc_1': 202, 'Loc_2': 204, 'Loc_3': 206}

# === Match BCI_script settings ===
SFREQ = 100
DECIM = 1                 # keep aligned with BCI_script.py (you can set 2 if desired)
N_FOLDS = 5
CV_RANDOM_STATE = 100

# RC parameters (match BCI_script.py)
N_RESERVOIR = 800
rc_params = dict(
    n_res=N_RESERVOIR,
    input_scaling=0.5,
    bias_scaling=0.5,
    spectral_radius=0.95,
    tau_mode=0.01,
    tau_sigma=0.8,
    tau_min=0.002,
    tau_max=0.08,
    bidirectional=True,
    merge_mode="product",
)

# ================= 2. Decoding functions =================
def run_lda_decoding_accuracy(epochs, name):
    """LDA baseline (time-resolved accuracy), similar style as BCI_script baseline."""
    print(f"    Running LDA ({name})...")
    X = epochs.get_data()  # (n_trials, n_ch, n_times)
    y = (epochs.events[:, 2] % 10) // 2  
    if DECIM > 1:
        X = X[:, :, ::DECIM]
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    slider = SlidingEstimator(clf, n_jobs=2, scoring='accuracy', verbose=False)
    scores = cross_val_multiscore(slider, X, y, cv=N_FOLDS)
    return np.mean(scores, axis=0)

def run_rc_decoding_heterorc(epochs, name):
    """HeteroRC + Ridge (time-resolved accuracy) via our shared function."""
    print(f"    Running HeteroRC ({name})...")
    X = epochs.get_data()  # (n_trials, n_ch, n_times)
    y = (epochs.events[:, 2] % 10) // 2
    times = epochs.times

    if DECIM > 1:
        X = X[:, :, ::DECIM]
        times = times[::DECIM]
    # Decision smoothing (applied inside time_resolved_decoding_heterorc on decision_function)
    FWHM_MS = 25.0
    EFFECTIVE_SFREQ = SFREQ / DECIM
    SIGMA_POINTS = (FWHM_MS / 1000.0) * EFFECTIVE_SFREQ / 2.355  # FWHM->sigma
    
    scores = time_resolved_decoding_heterorc(
        X=X,
        y=y,
        times=times,
        n_folds=N_FOLDS,
        fs=EFFECTIVE_SFREQ,          # important for dt inside reservoir (if used)
        rc_params=rc_params,
        scale_percentile=99,         # robust scaling on training set per fold
        cv_random_state=CV_RANDOM_STATE,
        metric = 'accuracy',
        rc_seed_mode="fixed",
        base_rc_random_state=100,    # consistent reservoir across folds (like BCI_script style)
        smooth_decisions=True,
        smooth_sigma_points=SIGMA_POINTS,
        verbose=False
    )
    return scores


# ================= 3. process_subject_and_save =================
def process_subject_and_save(file_path):
    sub_id = os.path.basename(file_path).replace(".bdf", "")
    save_path = os.path.join(RESULT_DIR, f"{sub_id}_decoding.npz")

    raw = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')

    # resample first (as you did), then pick channels
    raw.resample(SFREQ, npad='auto', verbose='ERROR')

    # Subject 7 status fix: Apply an 8-bit mask to the trigger channel to resolve 
    # a known hardware scaling issue specific to this subject in the original dataset.
    if "subject_7" in sub_id:
        print(f"  [Fix] Applying 8-bit mask to Status channel for {sub_id}")
        status_idx = raw.ch_names.index('Status')
        raw._data[status_idx] %= 256
        min_dur = 0.002
        events = mne.find_events(raw, stim_channel='Status', min_duration=min_dur, verbose='ERROR')
    else:
        events = mne.find_events(raw, stim_channel='Status', verbose='ERROR')

    # EEG picks 
    raw.pick_channels(raw.ch_names[:64])

    # --- Match BCI_script preprocessing for voltage branch ---
    raw_v = raw.copy()
    raw_v.filter(0.5, 30.0, fir_design='firwin', verbose='ERROR')

    # Epochs
    ep_v_p = mne.Epochs(
        raw_v, events, event_id_ping,
        tmin=-0.2, tmax=0.6,
        baseline=(-0.2, 0),
        preload=True, verbose='ERROR'
    )
    ep_v_n = mne.Epochs(
        raw_v, events, event_id_noping,
        tmin=-0.2, tmax=0.6,
        baseline=(-0.2, 0),
        preload=True, verbose='ERROR'
    )

    ep_v_p.equalize_event_counts(event_id_ping)
    ep_v_n.equalize_event_counts(event_id_noping)

    # LDA (accuracy) + HeteroRC (accuracy)
    s_v_lda_p = run_lda_decoding_accuracy(ep_v_p, "Volt-LDA Ping")
    s_v_lda_n = run_lda_decoding_accuracy(ep_v_n, "Volt-LDA NoPing")
    s_rc_p = run_rc_decoding_heterorc(ep_v_p, "HeteroRC Ping")
    s_rc_n = run_rc_decoding_heterorc(ep_v_n, "HeteroRC NoPing")

    # --- Power branch ---
    raw_p = raw.copy()
    raw_p.filter(3.0, 15.0, fir_design='firwin', verbose='ERROR')
    raw_p.apply_hilbert(envelope=True, verbose='ERROR')
    raw_p._data = raw_p._data ** 2
    raw_p.set_eeg_reference('average', projection=False, verbose='ERROR')

    ep_p_p = mne.Epochs(
        raw_p, events, event_id_ping,
        tmin=-0.2, tmax=0.6,
        baseline=(-0.2, 0),
        preload=True, verbose='ERROR'
    )
    ep_p_n = mne.Epochs(
        raw_p, events, event_id_noping,
        tmin=-0.2, tmax=0.6,
        baseline=(-0.2, 0),
        preload=True, verbose='ERROR'
    )

    ep_p_p.equalize_event_counts(event_id_ping)
    ep_p_n.equalize_event_counts(event_id_noping)

    s_p_lda_p = run_lda_decoding_accuracy(ep_p_p, "Power-LDA Ping")
    s_p_lda_n = run_lda_decoding_accuracy(ep_p_n, "Power-LDA NoPing")

    # save
    np.savez(
        save_path,
        times=ep_v_p.times[::DECIM] if DECIM > 1 else ep_v_p.times,
        v_lda_p=s_v_lda_p, v_lda_n=s_v_lda_n,
        p_lda_p=s_p_lda_p, p_lda_n=s_p_lda_n,
        rc_p=s_rc_p, rc_n=s_rc_n
    )

    return (ep_v_p.times[::DECIM] if DECIM > 1 else ep_v_p.times,
            s_v_lda_p, s_v_lda_n, s_p_lda_p, s_p_lda_n, s_rc_p, s_rc_n)


# ================= 4. parallel processing  =================
results = {'times': None, 'v_lda_p': [], 'v_lda_n': [], 'p_lda_p': [], 'p_lda_n': [], 'rc_p': [], 'rc_n': []}

def safe_process(f):
    try:
        return process_subject_and_save(f)
    except Exception as e:
        print(f"Error processing {f}: {e}")
        return None

n_jobs = 12  
all_res = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
    delayed(safe_process)(f) for f in subject_files
)


for res in all_res:
    if res is None:
        continue
    results['times'] = res[0]
    results['v_lda_p'].append(res[1]); results['v_lda_n'].append(res[2])
    results['p_lda_p'].append(res[3]); results['p_lda_n'].append(res[4])
    results['rc_p'].append(res[5]); results['rc_n'].append(res[6])