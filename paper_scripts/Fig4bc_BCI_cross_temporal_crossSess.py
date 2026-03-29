# -*- coding: utf-8 -*-
"""
Title:      BCI Competition IV 2a - CTG (HeteroRC Train(T)->Test(E) Cross-Generalisation + optional LDA baseline)
Date:       Jan 2026
Description:
    This script computes cross-temporal generalisation where the readout is trained on
    the training session (T) and evaluated on the testing session (E).

    Compared to the previous version:
    - RC part is changed from "within-session CV CTG" to "train(T)->test(E) CTG"
      using `cross_generalisation_train_test_heterorc`.
    - The optional LDA baseline (train(T)->test(E)) is kept.

Requirements:
    - heterorc.py must contain:
        1) class HeteroRC
        2) def cross_generalisation_train_test_heterorc(...)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import time
import mne
import numpy as np
import scipy.io

from sklearn.linear_model import RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

# --- New key function (must exist in your heterorc.py) ---
from heterorc import cross_generalisation_train_test_heterorc

# ================= Configuration =================
DATA_DIR = r'...'
SAVE_DIR = r'...'
os.makedirs(SAVE_DIR, exist_ok=True)

SUBJECTS = range(1, 10)
SFREQ = 100
N_RESERVOIR = 800
DECIM = 1
N_JOBS = 12

# --- RC parameters (do NOT include fs here; we pass fs explicitly; do include n_res) ---
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

# ========================= Data Helper Functions =========================
def add_montage(raw):
    """Rename channels and attach a standard 10-20 montage compatible with BCI2a."""
    bci2a_channels = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]
    orig_names = raw.ch_names[:22]
    rename_dict = dict(zip(orig_names, bci2a_channels))
    raw.rename_channels(rename_dict)
    for ch in raw.ch_names[22:]:
        raw.set_channel_types({ch: 'eog'})
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    return raw


def load_subject_data(sub_id):
    """
    Load BCI2a subject data.
    - Session T: provides X_train, y_train
    - Session E: provides X_test, y_test
    Returns:
        X_train, y_train, X_test, y_test, times
    """
    print(f"Loading Subject A0{sub_id}...", end="")

    def load_raw_simple(fname):
        fpath = os.path.join(DATA_DIR, fname)
        raw = mne.io.read_raw_gdf(fpath, preload=True, verbose=False)
        raw = add_montage(raw)
        raw.filter(0.5, 30., fir_design='firwin', verbose=False)
        raw.resample(SFREQ, npad="auto", verbose=False)
        raw.set_eeg_reference('average', projection=False, verbose=False)
        return raw

    # --- Training session (T) ---
    raw_t = load_raw_simple(f'A0{sub_id}T.gdf')
    events_t, _ = mne.events_from_annotations(
        raw_t,
        event_id={'769': 1, '770': 2, '771': 3, '772': 4},
        verbose=False
    )
    epochs_t = mne.Epochs(
        raw_t,
        events_t,
        tmin=-0.2,
        tmax=3.2,
        proj=False,
        picks='eeg',
        baseline=(-0.2, -0.05),
        preload=True,
        verbose=False
    )
    X_train = epochs_t.get_data(copy=True) * 1e6  # convert to microvolts
    y_train = epochs_t.events[:, -1]

    # --- Evaluation session (E) labels from .mat ---
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, f'A0{sub_id}E.mat'))
    y_test = np.concatenate([
        mat['data'][0][i]['y'][0, 0].flatten() if 'y' in mat['data'][0][i].dtype.names
        else mat['data'][0][i]['classlabel'][0, 0].flatten()
        for i in range(3, 9)
    ])

    # --- Evaluation session (E) ---
    raw_e = load_raw_simple(f'A0{sub_id}E.gdf')
    descs = raw_e.annotations.description

    # Determine trial start annotations
    if '783' in descs:
        events_e, _ = mne.events_from_annotations(raw_e, event_id={'783': 100}, verbose=False)
    elif '768' in descs:
        events_start, _ = mne.events_from_annotations(raw_e, event_id={'768': 100}, verbose=False)
        events_e = events_start.copy()
        events_e[:, 0] += int(2.0 * SFREQ)  # shift by 2 seconds
    else:
        raise RuntimeError("Could not find start markers ('783' or '768') in evaluation session annotations.")

    # Keep last 288 trials if extra markers exist
    if len(events_e) > 288:
        events_e = events_e[-288:]

    epochs_e = mne.Epochs(
        raw_e,
        events_e,
        tmin=-0.2,
        tmax=3.2,
        proj=False,
        picks='eeg',
        baseline=(-0.2, -0.05),
        preload=True,
        verbose=False
    )
    X_test = epochs_e.get_data(copy=True) * 1e6

    print(" Done.")
    return X_train, y_train, X_test, y_test, epochs_t.times


# ========================= Optional Baseline: LDA CTG (T -> E) =========================
def fit_and_score_tgm_smooth(X_train_t, y_train, X_test_full, y_test, sigma=None):
    """
    Train at a fixed train-time slice on session T, test across all test times on session E.

    Steps:
      1) Fit LDA on X_train_t.
      2) Compute decision scores across all test times in X_test_full.
      3) Optionally smooth decision scores along the time axis.
      4) Convert scores to predicted labels and compute accuracy vs y_test for each test time.

    Returns:
        accuracy_row: shape (n_test_times,)
    """
    clf = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )
    clf.fit(X_train_t, y_train)

    n_trials, _, n_test_times = X_test_full.shape
    classes = clf.classes_

    # Collect decision_function across all test times
    decisions_list = []
    for t in range(n_test_times):
        d = clf.decision_function(X_test_full[:, :, t])
        decisions_list.append(d)
    decisions = np.array(decisions_list)

    # Reformat to (trial, class, time) for multiclass; (trial, time) for binary
    if decisions.ndim == 3:
        decisions = decisions.transpose(1, 2, 0)  # (time, trial, class) -> (trial, class, time)
    else:
        decisions = decisions.T  # (time, trial) -> (trial, time)

    # Optional smoothing along time axis (last axis)
    if sigma is not None and sigma > 0:
        decisions = gaussian_filter1d(decisions, sigma=sigma, axis=-1, mode='nearest')

    # Convert decisions to predicted labels and compute accuracy over time
    if decisions.ndim == 3:
        preds_idx = np.argmax(decisions, axis=1)        # (trial, time)
        preds_labels = classes[preds_idx]
        accuracy_row = np.mean(preds_labels == y_test[:, None], axis=0)
    else:
        preds_idx = (decisions > 0).astype(int)
        preds_labels = classes[preds_idx]
        accuracy_row = np.mean(preds_labels == y_test[:, None], axis=0)

    return accuracy_row

# ================= Main Processing Loop (Parallel across subjects) =================
print(f"Starting Processing. Parallel Jobs (subjects): {N_JOBS}")

def process_one_subject(sub_id: int):
    save_path = os.path.join(SAVE_DIR, f'A0{sub_id}_results.npz')
    if os.path.exists(save_path):
        print(f"Skipping A0{sub_id} (already exists).")
        return None

    print(f"\nProcessing Subject A0{sub_id}...")
    X_train, y_train, X_test, y_test, times = load_subject_data(sub_id)

    # ---- decimation ----
    if DECIM is None or DECIM < 1:
        raise ValueError("DECIM must be an integer >= 1.")
    if DECIM > 1:
        X_train = X_train[:, :, ::DECIM]
        X_test  = X_test[:, :, ::DECIM]
        times   = times[::DECIM]
    times_ds = times
    print(f"  After decimation: n_times = {len(times_ds)} (DECIM={DECIM})")

    # ---- RC CTG ----
    print(f"  Computing RC CTG Train(T)->Test(E) ({len(times)}x{len(times)})...")
    t0 = time.time()

    FWHM_MS = 25.0
    EFFECTIVE_SFREQ = SFREQ / DECIM
    SIGMA_POINTS = (FWHM_MS / 1000.0 * EFFECTIVE_SFREQ) / 2.355

    tgm_rc = cross_generalisation_train_test_heterorc(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        times=times,
        fs=EFFECTIVE_SFREQ,
        rc_params=rc_params,
        scale_percentile=99,
        rc_random_state=42,
        smooth_decisions=True,
        smooth_sigma_points=SIGMA_POINTS,
        verbose=False, 
    )
    print(f"  RC CTG finished in {time.time() - t0:.1f}s.")

    # ---- LDA CTG baseline ----
    print(f"  Computing LDA CTG Train(T)->Test(E) ({X_train.shape[2]}x{X_train.shape[2]})...")
    t0 = time.time()

    tgm_lda_rows = [
        fit_and_score_tgm_smooth(
            X_train[:, :, t],
            y_train,
            X_test,
            y_test,
            sigma=None
        )
        for t in range(X_train.shape[2])
    ]
    tgm_lda = np.array(tgm_lda_rows)

    print(f"  LDA CTG finished in {time.time() - t0:.1f}s.")

    # ---- save ----
    np.savez(
        save_path,
        tgm_lda=tgm_lda,
        tgm_rc=tgm_rc,
        times=times_ds,
        subject_id=sub_id
    )
    print(f"  Saved to: {save_path}")
    return sub_id


out = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
    delayed(process_one_subject)(sub_id) for sub_id in SUBJECTS
)

print("\nAll processing complete!")

