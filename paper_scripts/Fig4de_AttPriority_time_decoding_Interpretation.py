# -*- coding: utf-8 -*-
"""
===============================================================================
Title:       Individual-Level Interpretation of Latent Dynamics (Figure 4d, e)
Author:      Runhao Lu
Date:        Mar 2026

Description:a
    End-to-end decoding and interpretation pipeline for the Attentional Priority 
    dataset. This script extracts individual-specific virtual source signals to 
    reveal the distinct latent dynamics driving classification in 'Ping' (impulse) 
    and 'No-Ping' (baseline) conditions.

Methodological Pipeline:
------------------------
1.  Strict Cross-Validation for Peak Selection: 
    A time-resolved cross-validated decoding curve is computed. The peak decoding 
    time is identified from this CV curve to completely prevent circular analysis 
    (selection bias) during subsequent interpretation.

2.  Full-Dataset Refitting for Interpretation: 
    To maximize the signal-to-noise ratio for physiological interpretation, a 
    single fixed HeteroRC reservoir is applied to the full dataset, and a Ridge 
    readout is fitted strictly at the identified peak time.

3.  Latent Dynamics Interpretation: 
    Applies Haufe's transform to convert backward readout weights into forward 
    activation patterns. The top 25 most informative units are clustered based 
    on their temporal dynamics to derive and visualize latent virtual sources.
===============================================================================
"""

import os
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt
import scipy.signal

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from heterorc_interpretation import analyze_dynamics
from heterorc import HeteroRC, time_resolved_decoding_heterorc


# =========================
# 1) PATHS & FILES
# =========================
BASE_DIR = r"..."
DATA_DIR = r"..."
RESULT_DIR = os.path.join(BASE_DIR, "...")
os.makedirs(RESULT_DIR, exist_ok=True)

subject_files = glob.glob(os.path.join(DATA_DIR, "subject_*.bdf"))

# =========================
# 2) EVENTS
# =========================
event_id_ping = {'Loc_0': 100, 'Loc_1': 102, 'Loc_2': 104, 'Loc_3': 106}
event_id_noping = {'Loc_0': 200, 'Loc_1': 202, 'Loc_2': 204, 'Loc_3': 206}
class_names = ["Loc_1", "Loc_2", "Loc_3", "Loc_4"]
# =========================
# 3) GLOBAL SETTINGS
# =========================
dsp_rate = 100  # Hz
SFREQ = 100     # effective sampling rate after resample
N_RESERVOIR = 800

# Keep RC params explicit (avoid hidden defaults)
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


# =========================
# DECODING SETTINGS
# =========================
N_FOLDS = 5
SCALE_PERCENTILE = 99

# If you want smoother curves:
#   set smooth_decisions=True, and pick sigma_points.
# Here we keep legacy behavior (no smoothing).
SMOOTH_DECISIONS = False
SMOOTH_SIGMA_POINTS = 2.0
SMOOTH_MODE = "nearest"


# =============================================================================
# 4) DATA LOADING / PREPROCESSING
# =============================================================================
def load_and_preprocess_bdf(file_path: str):
    """
    Load one BDF file, fix trigger issues, keep first 64 channels, filter 0.5-30 Hz.
    Returns:
        raw_eeg (Raw), events (ndarray)
    """
    raw = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')
    raw.resample(dsp_rate, npad='auto')

    # Try montage (optional, best-effort)
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
    except Exception:
        pass

    # Trigger fix for subject_7 
    sub_id = os.path.basename(file_path)
    if "subject_7" in sub_id:
        status_idx = raw.ch_names.index('Status')
        raw._data[status_idx] %= 256
        min_dur = 0.002
    else:
        min_dur = 1 / raw.info['sfreq']

    events = mne.find_events(raw, stim_channel='Status', min_duration=min_dur, verbose='ERROR')

    # Keep first 64 channels (EEG)
    raw.pick_channels(raw.ch_names[:64])

    # Band-pass for voltage branch
    raw.filter(0.5, 30.0, verbose='ERROR')

    return raw, events


def make_epochs(raw: mne.io.BaseRaw, events: np.ndarray, eid: dict, tmin=-0.2, tmax=0.6):
    """
    Create epochs for one condition (eid = event_id_ping or event_id_noping)
    and compute 4-class labels:
        y = (event_code % 10) // 2 -> 0..3
    """
    epochs = mne.Epochs(
        raw, events, eid,
        tmin=tmin, tmax=tmax,
        baseline=None,
        preload=True,
        verbose='ERROR'
    )
    # map trigger codes to 0..3 
    y = (epochs.events[:, 2] % 10) // 2
    return epochs, y

# =============================================================================
# 5) DECODING: use heterorc.time_resolved_decoding_heterorc
# =============================================================================
def run_rc_decoding_cv(epochs: mne.Epochs, y: np.ndarray, name: str):
    """
    Time-resolved CV decoding curve using HeteroRC features + Ridge readout.
    Output: Dec Acc over time, macro-average OvR for 4 classes.
    """
    print(f"    Running RC decoding (CV) for: {name}")

    X = epochs.get_data()      # (n_trials, n_chans, n_times)
    times = epochs.times

    auc_curve = time_resolved_decoding_heterorc(
        X=X, y=y, times=times,
        n_folds=N_FOLDS,
        fs=SFREQ,
        rc_params=rc_params,
        scale_percentile=SCALE_PERCENTILE,
        rc_seed_mode="fixed",  
        metric="accuracy",
        smooth_decisions=SMOOTH_DECISIONS,
        smooth_sigma_points=SMOOTH_SIGMA_POINTS,
        smooth_mode=SMOOTH_MODE,
        verbose=False,
    )
    return auc_curve


# =============================================================================
# 6) INTERPRETATION PIPELINE (UPDATED)
# =============================================================================

def run_interpretation_pipeline(file_path: str, condition: str = "noping", top_n: int = 25, n_clusters: int = 3):
    """
    Interpretation steps (single subject, single condition):
    1) Load + preprocess raw
    2) Make epochs for selected condition
    3) Compute CV decoding curve over time using heterorc.time_resolved_decoding_heterorc
    4) Pick peak time
    5) Build ONE fixed reservoir on FULL dataset, fit readout at peak time
    6) Stage1 + Stage2 (using heterorc_interpretation module)
    """
    print("\n" + "#" * 60)
    print("STARTING INTERPRETATION PIPELINE (UPDATED)")
    print("#" * 60)
    print(f"File: {os.path.basename(file_path)} | Condition: {condition}")
    base = os.path.basename(file_path)
    sub_id = base.replace(".bdf", "")

    raw, events = load_and_preprocess_bdf(file_path)

    # Choose condition
    if condition.lower() == "ping":
        eid = event_id_ping
        cond_name = "PING"
    else:
        eid = event_id_noping
        cond_name = "NOPING"

    epochs, y = make_epochs(raw, events, eid, tmin=-0.2, tmax=0.6)
    X = epochs.get_data()
    times = epochs.times
    
    # ---------- 1) Peak search ----------
    print("-> Computing CV decoding curve (peak search) ...")
    decoding_curve = time_resolved_decoding_heterorc(
        X=X,
        y=y,
        times=times,
        n_folds=N_FOLDS,
        fs=SFREQ,
        rc_params=rc_params,
        scale_percentile=SCALE_PERCENTILE,
        rc_seed_mode="fixed",
        metric="accuracy",
        smooth_decisions=SMOOTH_DECISIONS,
        smooth_sigma_points=SMOOTH_SIGMA_POINTS,
        smooth_mode=SMOOTH_MODE,
        verbose=False,
    )

    peak_idx = int(np.argmax(decoding_curve))
    peak_time = float(times[peak_idx])
    peak_auc = float(decoding_curve[peak_idx])
    print(f"-> Peak found: t={peak_time:.3f}s | CV-decoding={peak_auc:.3f} ({cond_name})")

    # Plot curve
    plt.figure(figsize=(8, 4))
    plt.plot(times, decoding_curve, lw=2, label=f"RC (CV-decoding) - {cond_name}")
    plt.axhline(0.25, color='gray', linestyle=':', label='Chance (0.25)')
    plt.axvline(0.0, color='k', linewidth=1, alpha=0.6)
    plt.scatter([peak_time], [peak_auc], s=60, zorder=5)
    plt.xlabel("Time (s)")
    plt.ylabel("Decoding")
    plt.title(f"Peak search - {os.path.basename(file_path)}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- 2) Refit on FULL data at peak time (for Haufe & interpretation) ----------
    print("-> Refitting a single model on FULL data at peak time ...")

    # Robust scaling on FULL data (post-hoc interpretation)
    scale_val = np.percentile(np.abs(X), SCALE_PERCENTILE)
    if not np.isfinite(scale_val) or scale_val == 0:
        scale_val = 1.0
    X_s = X / scale_val

    # One fixed reservoir (choose a fixed seed)
    esn = HeteroRC(
        n_in=X.shape[1],
        fs=SFREQ,
        random_state=42,
        **rc_params
    )
    S = esn.transform(X_s)  # (n_trials, n_res, n_times)

    # Fit readout at peak time
    clf = make_pipeline(
        StandardScaler(),
        RidgeClassifierCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0,1000.0))
    )
    clf.fit(S[:, :, peak_idx], y)

    # ---------- 3) Stage 1: Dynamics (Clustering -> Virtual Sources) ----------
    phase_name = f"{cond_name}_Peak{peak_time*1000:.0f}ms"
    
    # NEW: Calling the modularized function with full parameter control
    stage1  = analyze_dynamics(
        esn=esn,
        classifier=clf,
        target_time=peak_time,
        state_snapshot=S,
        y_labels=y,
        times=times,
        n_clusters=n_clusters,
        top_n=top_n,
        phase_name=phase_name,
        
        # --- Visualization Parameters ---
        # ERP: Subtract mean of baseline period (-0.2 to 0.0)
        erp_range=(-0.2, 0.6),
        erp_baseline_mode="mean",
        erp_baseline_range=(-0.2, 0.0),
        
        # TFR: Use dB conversion (logratio) relative to baseline
        tfr_baseline_mode="logratio",
        tfr_baseline_range=(-0.2, 0.0),
        tfr_freqs=np.arange(2, 35, 1),  # Finer frequency steps
        
        # FOOOF: Standard settings
        fooof_params={'max_n_peaks': 4, 'aperiodic_mode': 'fixed'},
       
        class_names=class_names,
        return_results=True,  
    
        plot_style="poster",  # "poster", "paper", or None
        figsize=(18,9),  # 2 cluste = 8
        
        raw_X_snapshot=X,          # NEW
        info=epochs.info,          # NEW
        inline_topomaps=True,      # NEW
        cov_window_half_width=0.1, # same as before stage2
    )
    
    
    fig = stage1["figure"]
    
    # User-defined naming (subject, condition, peak time, etc.)
    fname = f"{sub_id}_{cond_name}_Peak{peak_time*1000:.0f}ms.png"
    fig.savefig(
        os.path.join(RESULT_DIR, fname),
        dpi=500,
        bbox_inches="tight"
    )



# =============================================================================
# 10) MAIN
# =============================================================================
subject_files=subject_files[15:16]
if __name__ == "__main__":
    for i,fp in enumerate(subject_files):
        run_interpretation_pipeline(fp, condition="ping", top_n=25, n_clusters=2)
   

