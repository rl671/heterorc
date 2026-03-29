# -*- coding: utf-8 -*-
"""
===============================================================================
Title:       Group-Level Interpretation via Sensor-Space Matching (Figure 5b, c)
Author:      Runhao Lu
Date:        Mar 2026

Description:
    Group-level interpretation pipeline for the Attentional Priority dataset. 
    This script implements a "sensor-space matching" algorithm to bridge the 
    idiosyncratic latent reservoir spaces across independent participants, 
    revealing distinct population-level motifs for 'Ping' and 'No-Ping' trials.

Methodological Pipeline:
------------------------
1.  Subject-Level Feature Extraction: 
    Identifies the peak decoding time via cross-validation, fits the readout, 
    and extracts the top latent reservoir units for each participant.
2.  Sensor-Space Projection & Sign-Alignment: 
    Projects the most informative units back to the common physical EEG sensor 
    space. Polarity is strictly aligned (forcing the maximum pole positive) 
    to prevent destructive signal cancellation during group averaging.
3.  Global Spatial Clustering & Reconstruction: 
    Clusters the pooled spatial topographies globally and maps the cluster 
    assignments back to individual reservoirs to reconstruct grand-average 
    ERPs, TFRs, PSDs, and Topomaps.
===============================================================================
"""

import os
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from mne.time_frequency import tfr_array_morlet

from heterorc import HeteroRC, time_resolved_decoding_heterorc
from heterorc_interpretation import analyze_dynamics_group
# =========================
# 1) CONFIGURATION
# =========================
BASE_DIR = r"C:\Users\Lenovo\Desktop\RC\Results_AttPri"
DATA_DIR = r"D:\RC_proj\Duncan2023\Data"
RESULT_DIR = os.path.join(BASE_DIR, "Group_Interpretation_Aligned_RH")
os.makedirs(RESULT_DIR, exist_ok=True)

subject_files = glob.glob(os.path.join(DATA_DIR, "subject_*.bdf"))

CONDITION = "noping"  # "ping" or "noping"
TOP_N = 10           # Reduced to Top 5 for cleaner group-level motifs
N_CLUSTERS = 2
SFREQ = 100
COV_WINDOW = 0.1 

rc_params = dict(
    n_res=800, input_scaling=0.5, bias_scaling=0.5, spectral_radius=0.95,
    tau_mode=0.01, tau_sigma=0.8, tau_min=0.002, tau_max=0.08,
    bidirectional=True, merge_mode="product"
)

# =========================
# 2) HELPER FUNCTIONS
# =========================
def load_data(file_path, condition):
    raw = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')
    raw.resample(SFREQ, npad='auto')
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
    except: pass
    # Subject 7 hardware fix: Apply an 8-bit mask to the trigger channel to resolve 
    # a known event-marker scaling issue specific to this subject in the raw dataset.
    if "subject_7" in os.path.basename(file_path):
        status_idx = raw.ch_names.index('Status')
        raw._data[status_idx] %= 256
        min_dur = 0.002
    else: min_dur = 1 / raw.info['sfreq']

    events = mne.find_events(raw, stim_channel='Status', min_duration=min_dur, verbose='ERROR')
    raw.pick_channels(raw.ch_names[:64])
    raw.filter(0.5, 30.0, verbose='ERROR')

    eid = {'Loc_0': 100, 'Loc_1': 102, 'Loc_2': 104, 'Loc_3': 106} if condition == "ping" else {'Loc_0': 200, 'Loc_1': 202, 'Loc_2': 204, 'Loc_3': 206}
    epochs = mne.Epochs(raw, events, eid, tmin=-0.2, tmax=0.6, baseline=None, preload=True, verbose='ERROR')
    y = (epochs.events[:, 2] % 10) // 2
    return raw, epochs, y

def extract_spatial_topo_aligned(X, S_unit, times, peak_time, window=0.1):
    """ Extract 64-channel spatial topography AND align the sign. """
    t_min, t_max = peak_time - window, peak_time + window
    win_mask = (times >= t_min) & (times <= t_max)
    
    X_flat = X[:, :, win_mask].transpose(1, 0, 2).reshape(X.shape[1], -1)
    S_flat = S_unit[:, win_mask].reshape(-1)
    
    X_flat -= np.mean(X_flat, axis=1, keepdims=True)
    S_flat -= np.mean(S_flat)
    
    cov_map = (X_flat @ S_flat) / (len(S_flat) - 1)
    
    # SIGN ALIGNMENT
    max_idx = np.argmax(np.abs(cov_map))
    sign_flip = 1.0 if cov_map[max_idx] > 0 else -1.0
    cov_map_aligned = cov_map * sign_flip
    
    cov_z = (cov_map_aligned - np.mean(cov_map_aligned)) / (np.std(cov_map_aligned) + 1e-15)
    return cov_z, sign_flip

# =========================
# 3) PHASE 1: EXTRACT TOPOS & SIGNS
# =========================
print(f"--- PHASE 1: Extracting Spatial Topographies for Top {TOP_N} units (Sign-Aligned) ---")
all_sub_data = []
global_spatial_features = []
info = None

for fp in subject_files[6:]:
    sub_id = os.path.basename(fp).replace('.bdf', '')
    print(f"Processing {sub_id}...")
    
    raw, epochs, y = load_data(fp, CONDITION)
    X = epochs.get_data()
    times = epochs.times
    if info is None:
        info = epochs.info
    auc = time_resolved_decoding_heterorc(X, y, times, n_folds=5, fs=SFREQ, rc_params=rc_params, rc_seed_mode="fixed", metric="accuracy", verbose=False)
    peak_idx = int(np.argmax(auc))
    peak_time = times[peak_idx]
    
    scale_val = np.percentile(np.abs(X), 99)
    X_s = X / (scale_val if scale_val>0 else 1.0)
    esn = HeteroRC(n_in=X.shape[1], fs=SFREQ, random_state=42, **rc_params)
    S = esn.transform(X_s)
    
    clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=(0.01, 0.1, 1.0, 10.0)))
    clf.fit(S[:, :, peak_idx], y)
    # scores = clf.decision_function(S[:, :, peak_idx]) if hasattr(clf, "decision_function") else clf.predict_proba(S[:, :, peak_idx])
    # Pack extracted features, raw data, and trained model instances.
    # This list serves as the input pool for the global sensor-space matching 
    # and clustering algorithm executed by analyze_dynamics_group().
    all_sub_data.append({
        'X': X,
        'S': S,
        'y': y,
        'times': times,
        'target_time': peak_time,
        'classifier': clf,
        'esn': esn
    })
    
results = analyze_dynamics_group(
    subject_data_list=all_sub_data,
    info=info,
    return_results=True,
    export_virtual_sources=True,
    figsize=(18,9)
)
fig = results["figure"]

fname = f"GrandAverage_{CONDITION}_Top{TOP_N}_cluster{N_CLUSTERS}.png"
fig.savefig(
    os.path.join(RESULT_DIR, fname),
    dpi=500,
    bbox_inches="tight"
)
