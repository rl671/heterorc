"""
===============================================================================
Title:       Individual-Level Interpretation of Reservoir Dynamics (Figure 3d)
Author:      Runhao Lu
Date:        Jan 2026

Description:
    End-to-end decoding and interpretation pipeline for the BCI Competition IV-2a 
    dataset. This script extracts individual-specific latent virtual source signals
    to reveal the spectrotemporal and spatial neural motifs driving classification.

Methodological Pipeline:
------------------------
1.  Strict Cross-Validation for Peak Selection: 
    A time-resolved cross-validated decoding curve is computed exclusively on the 
    training session (A0xT). The peak decoding time is identified from this curve 
    to completely prevent circular analysis (selection bias) on the evaluation set.

2.  Full-Dataset Refitting for Interpretation: 
    To maximize signal-to-noise ratio for physiological interpretation, the data 
    from both sessions (Train + Eval) are concatenated. A fixed HeteroRC reservoir 
    is applied, and a Ridge readout is fitted strictly at the identified peak time.

3.  Latent Dynamics Interpretation: 
    Applies Haufe's transform to convert backward readout weights into forward 
    activation patterns. The most informative units are clustered based on their 
    temporal dynamics to derive latent virtual source signals. These sources are 
    then evaluated across temporal (ERP), spectral (TFR/PSD), and spatial (Topomaps) domains.
===============================================================================
"""

import os

# -----------------------------------------------------------------------------
# Environment variables to reduce multi-threading interference (optional)
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

from heterorc import (
    HeteroRC,
    time_resolved_decoding_heterorc,                 # CV peak search
    time_resolved_decoding_train_test_heterorc       # Optional reporting curve
)
from heterorc_interpretation import analyze_dynamics

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = r"D:\RC_proj\BCI2a\Data"
SAVE_DIR = r"C:\Users\Lenovo\Desktop\RC\Results_BCI2a\Results_interpretation"
os.makedirs(SAVE_DIR, exist_ok=True)

SUBJECT = 1

SFREQ = 100
DECIM = 1  # If >1, decimate epochs in time (after resampling to SFREQ)
EFFECTIVE_SFREQ = SFREQ / DECIM

N_RESERVOIR = 800

# Peak search CV folds (training session only)
N_FOLDS = 5

# HeteroRC parameters (bidirectional + product merge)
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

ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SCALE_PERCENTILE = 99
RC_RANDOM_STATE = 42

# Optional temporal smoothing on decision function
FWHM_MS = 25.0
SIGMA_POINTS = (FWHM_MS / 1000.0 * EFFECTIVE_SFREQ) / 2.355

# Analysis windows (in seconds, relative to cue/epoch t=0)
PHASES = {
    "Cue": (0.0, 1.25),
    "Imagination": (1.25, 3.0),
}
bci2a_class_names = ["LH", "RH", "Foot", "Tongue"]

# =============================================================================
# Helper: Montage for BCI2a
# =============================================================================
def add_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Rename the first 22 EEG channels to known BCI2a labels and set montage.
    Remaining channels are treated as EOG.
    """
    bci2a_channels = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz"
    ]

    orig_names = raw.ch_names[:22]
    rename_dict = dict(zip(orig_names, bci2a_channels))
    raw.rename_channels(rename_dict)

    # Remaining channels: treat as EOG (depending on dataset variant)
    for ch in raw.ch_names[22:]:
        raw.set_channel_types({ch: "eog"})

    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
    return raw

# =============================================================================
# Data loading
# =============================================================================
def load_subject_data(sub_id: int):
    """
    Load BCI2a subject training session (A0xT.gdf) and evaluation session (A0xE.gdf),
    plus evaluation labels from A0xE.mat.

    Returns
    -------
    X_train : ndarray, shape (n_train, n_ch, n_times)
    y_train : ndarray, shape (n_train,)
    X_test  : ndarray, shape (n_test,  n_ch, n_times)
    y_test  : ndarray, shape (n_test,)
    times   : ndarray, shape (n_times,)
    info    : mne.Info
    """
    print(f"Loading Subject A0{sub_id}... ", end="")

    def load_raw_simple(fname: str) -> mne.io.BaseRaw:
        fpath = os.path.join(DATA_DIR, fname)
        raw = mne.io.read_raw_gdf(fpath, preload=True, verbose=False)
        raw = add_montage(raw)

        # Basic preprocessing (you can adjust as needed)
        raw.filter(0.5, 30.0, fir_design="firwin", verbose=False)
        raw.resample(SFREQ, npad="auto", verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        return raw

    # -------------------------
    # Training session (T)
    # -------------------------
    raw_t = load_raw_simple(f"A0{sub_id}T.gdf")
    events_t, _ = mne.events_from_annotations(
        raw_t, event_id={"769": 1, "770": 2, "771": 3, "772": 4}, verbose=False
    )

    epochs_t = mne.Epochs(
        raw_t, events_t,
        tmin=-0.2, tmax=3.2,
        proj=False, picks="eeg",
        baseline=(-0.2, 0),
        preload=True, verbose=False
    )

    X_train = epochs_t.get_data(copy=True) * 1e6  # convert to microvolts
    y_train = epochs_t.events[:, -1]

    # -------------------------
    # Evaluation labels (MAT)
    # -------------------------
    mat_path = os.path.join(DATA_DIR, f"A0{sub_id}E.mat")
    mat = scipy.io.loadmat(mat_path)

    # Common structure: concatenate across runs
    y_test = np.concatenate([
        mat["data"][0][i]["y"][0, 0].flatten()
        if "y" in mat["data"][0][i].dtype.names
        else mat["data"][0][i]["classlabel"][0, 0].flatten()
        for i in range(3, 9)
    ])

    # -------------------------
    # Evaluation session (E)
    # -------------------------
    raw_e = load_raw_simple(f"A0{sub_id}E.gdf")
    descs = raw_e.annotations.description

    # Try cue marker first ('783'), otherwise trial-start ('768') + shift
    if "783" in descs:
        events_e, _ = mne.events_from_annotations(
            raw_e, event_id={"783": 100}, verbose=False
        )
    elif "768" in descs:
        events_start, _ = mne.events_from_annotations(
            raw_e, event_id={"768": 100}, verbose=False
        )
        events_e = events_start.copy()
        events_e[:, 0] += int(2.0 * SFREQ)  # shift by 2 seconds
    else:
        raise RuntimeError("Could not find start markers ('783' or '768') in eval session.")

    # Some BCI2a files contain extra markers; keep last 288 if needed
    if len(events_e) > 288:
        events_e = events_e[-288:]

    epochs_e = mne.Epochs(
        raw_e, events_e,
        tmin=-0.2, tmax=3.2,
        proj=False, picks="eeg",
        baseline=(-0.2, 0),
        preload=True, verbose=False
    )

    X_test = epochs_e.get_data(copy=True) * 1e6

    # Basic sanity checks
    if len(X_test) != len(y_test):
        print("\nWARNING: Eval epoch count and y_test length mismatch.")
        print(f"  epochs_e: {len(X_test)} | y_test: {len(y_test)}")
        m = min(len(X_test), len(y_test))
        X_test = X_test[:m]
        y_test = y_test[:m]

    print("Done.")
    return X_train, y_train, X_test, y_test, epochs_t.times, epochs_t.info

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1) Load data
    # -------------------------------------------------------------------------
    X_train, y_train, X_test, y_test, times, info = load_subject_data(SUBJECT)

    # Optional temporal decimation (after resampling)
    if DECIM > 1:
        X_train = X_train[:, :, ::DECIM]
        X_test = X_test[:, :, ::DECIM]
        times = times[::DECIM]

    print(f"\n[Data Info]")
    print(f"  Times: {times[0]:.3f}s to {times[-1]:.3f}s | n_times={len(times)}")
    print(f"  Train: {X_train.shape[0]} trials | Test: {X_test.shape[0]} trials")
    print(f"  Classes (train): {np.unique(y_train)}")

    # -------------------------------------------------------------------------
    # 2) Peak search (TRAINING SET CV ONLY)
    # -------------------------------------------------------------------------
    print("\n[Step A] Computing TRAIN-CV decoding curve for peak search...")

    cv_curve = time_resolved_decoding_heterorc(
        X=X_train,
        y=y_train,
        times=times,
        n_folds=N_FOLDS,
        fs=EFFECTIVE_SFREQ,
        rc_params=rc_params,
        scale_percentile=SCALE_PERCENTILE,
        rc_seed_mode="fixed",               # fixed reservoir for all folds
        base_rc_random_state=RC_RANDOM_STATE,
        metric="accuracy",
        smooth_decisions=True,
        smooth_sigma_points=SIGMA_POINTS,
        smooth_mode="nearest",
        verbose=True,
    )

    # Plot CV curve
    plt.figure(figsize=(6, 3))
    plt.plot(times, cv_curve, lw=2, color="k")
    plt.axhline(0.25, linestyle="--", color="gray", alpha=0.6)
    plt.axvline(0.0, color="k", lw=1, alpha=0.5)
    plt.axvline(1.25, color="gray", linestyle=":", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Decoding accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"A0{SUBJECT}_TrainCV_PeakSearch.png"), dpi=300)
    plt.show()

    # # OPTIONAL: Compute train->test curve for reporting (NOT used for peak selection)
    # print("\n[Optional] Computing Train->Test decoding curve for reporting...")
    # tt_curve = time_resolved_decoding_train_test_heterorc(
    #     X_train, y_train, X_test, y_test, times,
    #     fs=EFFECTIVE_SFREQ,
    #     rc_params=rc_params,
    #     scale_percentile=SCALE_PERCENTILE,
    #     alphas=ALPHAS,
    #     rc_random_state=RC_RANDOM_STATE,
    #     smooth_decisions=True,
    #     smooth_sigma_points=SIGMA_POINTS,
    #     verbose=False
    # )

    # plt.figure(figsize=(10, 4))
    # plt.plot(times, tt_curve, lw=2, color="k", label="Train->Test (report only)")
    # plt.axhline(0.25, linestyle="--", color="gray", alpha=0.6, label="Chance (0.25)")
    # plt.axvline(0.0, color="k", lw=1, alpha=0.5)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Decoding accuracy")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(SAVE_DIR, f"A0{SUBJECT}_TrainTest_ReportCurve.png"), dpi=300)
    # plt.show()

    # -------------------------------------------------------------------------
    # 3) Build FULL dataset for interpretation (Train + Eval labels)
    # -------------------------------------------------------------------------
    print("\n[Step B] Building FULL dataset (Train + Eval) for interpretation...")

    # Ensure time dimension and channel dimension match
    if X_train.shape[1] != X_test.shape[1] or X_train.shape[2] != X_test.shape[2]:
        raise RuntimeError(
            "Train and Test epochs do not match in channels or time samples. "
            "Check preprocessing / event alignment / decimation settings."
        )

    X_full = np.concatenate([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)
    
    # Apply global robust scaling (99th percentile) across the concatenated dataset.
    # Note: Unlike the strictly isolated scaling used during cross-validation (Step A),
    # global scaling here ensures all trials occupy the same dynamic range, which is 
    # crucial for stable Haufe transform patterns and accurate virtual source reconstruction.
    scale_val = np.percentile(np.abs(X_full), SCALE_PERCENTILE)
    if (not np.isfinite(scale_val)) or scale_val == 0:
        scale_val = 1.0
    X_full_s = X_full / scale_val

    # One fixed reservoir (seed fixed for reproducibility)
    esn = HeteroRC(
        n_in=X_full.shape[1],
        random_state=RC_RANDOM_STATE,
        fs=EFFECTIVE_SFREQ,
        **rc_params
    )

    # Transform full dataset -> reservoir states
    S_full = esn.transform(X_full_s)  # (n_full, n_res, n_times)

    # -------------------------------------------------------------------------
    # 4) Interpretation at phase-specific peak times (peaks from TRAIN CV curve)
    # -------------------------------------------------------------------------
    print("\n[Step C] Running interpretation at TRAIN-CV peak times...")

    for phase_name, (t_min, t_max) in PHASES.items():
        # Find indices in the phase window
        mask = (times >= t_min) & (times <= t_max)
        if not np.any(mask):
            print(f"WARNING: No time points found in phase window: {phase_name}")
            continue

        window_indices = np.where(mask)[0]

        # Peak selection uses TRAIN CV curve
        peak_offset = int(np.argmax(cv_curve[window_indices]))
        peak_idx = int(window_indices[peak_offset])
        peak_time = float(times[peak_idx])
        peak_cv_acc = float(cv_curve[peak_idx])
        # peak_tt_acc = float(tt_curve[peak_idx])

        print("\n" + "-" * 72)
        print(f"Phase: {phase_name}")
        print(f"  Peak time (from Train CV): {peak_time:.3f}s")
        print(f"  Train-CV accuracy at peak: {peak_cv_acc:.3f}")
        # print(f"  Train->Test accuracy at peak (report): {peak_tt_acc:.3f}")

        # ---------------------------------------------------------------------
        # A) Fit readout on FULL data at peak time (post-hoc interpretation model)
        # ---------------------------------------------------------------------
        clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=ALPHAS))
        clf.fit(S_full[:, :, peak_idx], y_full)

        # ---------------------------------------------------------------------
        # B) Stage 1: Reservoir dynamics interpretation (Haufe -> clusters -> virtual sources)
        # ---------------------------------------------------------------------
        rc_interpretation  = analyze_dynamics(
            esn=esn,
            classifier=clf,
            target_time=peak_time,
            state_snapshot=S_full,
            y_labels=y_full,
            times=times,
            n_clusters=3,
            top_n=25,
            phase_name=f"A0{SUBJECT}_{phase_name}",
            # Visualization settings
            plot_style="poster",

            # ERP settings
            erp_range=(-0.2, 3.0),
            erp_baseline_mode="mean",
            erp_baseline_range=(-0.2, 0),

            # TFR settings (baseline in pre-cue interval)
            tfr_baseline_mode="logratio",
            tfr_baseline_range=(-0.2, 0.0),
            tfr_freqs=np.arange(2, 35, 1),

            # FOOOF settings
            fooof_params={"max_n_peaks": 4, "peak_width_limits": [1, 8]},
            figsize=(18, 12),
            
            class_names=bci2a_class_names, 
            return_results=True,
            
            raw_X_snapshot=X_full,          # NEW
            info=info,          # NEW
            inline_topomaps=True,      # NEW
            cov_window_half_width=0.1, # same as before stage2
        )
        
        # # Access virtual source
        Svirt_1 = rc_interpretation["clusters"]["Cluster 1"]["S_virt"]  # trial * times
        # # # Access neuron importance
        # importance = stage1["neurons"]["neuron_importance"]
        fig = rc_interpretation["figure"]
        
        # User-defined naming (subject, condition, peak time, etc.)
        fname = f"A0{SUBJECT}_Peak{peak_time*1000:.0f}ms.png"
        fig.savefig(
            os.path.join(SAVE_DIR, fname),
            dpi=500,
            bbox_inches="tight"
        )

    print("\nAll interpretation tasks completed.")
