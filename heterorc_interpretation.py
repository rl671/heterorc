# -*- coding: utf-8 -*-
"""
HeteroRC Interpretation Module
==============================

This module provides a comprehensive suite for interpreting the internal dynamics 
of a Heterogeneous Reservoir Computing (HeteroRC) model, specifically designed 
for EEG/MEG analysis.

It moves beyond "black box" prediction by converting reservoir states back into 
interpretable physiological domains using the Haufe Transform and Clustering.

Key Features:
-------------
1. **Haufe Transform**: Converts uninterpretable classifier weights (backward model) 
   into activation patterns (forward model), identifying which reservoir units 
   encode class-discriminative information.
2. **Ward Clustering**: Groups the most important reservoir units based on their 
   temporal dynamics, identifying functional "sub-networks" within the reservoir.
3. **Virtual Sources**: Constructs a representative time-series signal for each 
   cluster, effectively acting as a non-linear source separation algorithm.
4. **Multi-View Visualization**: Generates a composite figure for each virtual source:
   - **ERP**: Event-Related Potentials (Time domain).
   - **TFR**: Time-Frequency Representations (Time-Frequency domain).
   - **PSD**: Power Spectral Density with FOOOF parameterization (Frequency domain).
   - **Topomap**: Sensor-space projection of the virtual source (Spatial domain).

Dependencies:
-------------
- numpy, matplotlib, scipy
- mne (for Morlet wavelets and topomaps)
- sklearn (for scaling)
- fooof (optional, for aperiodic spectral parameterization)

Usage:
------
    from heterorc_interpretation import analyze_dynamics
    
    results = analyze_dynamics(
        esn=my_esn_model,
        classifier=my_readout_layer,
        target_time=0.35,  # Time of peak accuracy
        state_snapshot=states_matrix,
        y_labels=ground_truth,
        times=epoch_times,
        inline_topomaps=True,
        info=mne_info,
        raw_X_snapshot=raw_eeg_data
    )

New - group-level interpretation.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import scipy.signal
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from mne.time_frequency import tfr_array_morlet
from mne.baseline import rescale
import mne
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import (
    MaxNLocator,
    FormatStrFormatter,
    ScalarFormatter,
)
# =============================================================================
# HELPER: Plotting Styles
# =============================================================================
def _get_plot_settings(style="poster"):
    """
    Returns a dictionary of plot settings (font sizes, line widths, marker sizes)
    tailored for specific output formats.

    Parameters
    ----------
    style : str, optional
        One of 'poster', 'paper', or 'default'. Defaults to "poster".
    """

    if style == "poster":
        return {
            "font_size": 15,
            "title_size": 15,
            "label_size": 15,
            "tick_size": 15,
            "legend_size": 12,
            "line_width": 1.5,
            "scatter_s_bg": 50,
            "scatter_s_fg": 200,
            "scatter_lw": 2.0,
        }
    elif style == "paper":
        return {
            "font_size": 10,
            "title_size": 12,
            "label_size": 11,
            "tick_size": 9,
            "legend_size": 9,
            "line_width": 1.0,
            "scatter_s_bg": 15,
            "scatter_s_fg": 100,
            "scatter_lw": 1.0,
        }
    else: 
        return {
            "font_size": 12,
            "title_size": 14,
            "label_size": 12,
            "tick_size": 10,
            "legend_size": 10,
            "line_width": 1.5,
            "scatter_s_bg": 20,
            "scatter_s_fg": 150,
            "scatter_lw": 1.5,
        }

def _apply_rc_params(settings):
    plt.rcParams.update({
        'font.size': settings['font_size'],
        'axes.titlesize': settings['title_size'],
        'axes.labelsize': settings['label_size'],
        'xtick.labelsize': settings['tick_size'],
        'ytick.labelsize': settings['tick_size'],
        'legend.fontsize': settings['legend_size'],
        'lines.linewidth': settings['line_width'],
        'figure.autolayout': True,
        'font.weight': 'normal',
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
    })

def _set_axis_format(ax, axis='y', style='sci'):
    """
    Standardizes axis formatting.
    """
    if axis == 'x' or axis == 'both':
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    if axis == 'y' or axis == 'both':
        if style == 'sci':
            # 1. Locator: Clean integer-like steps
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            
            # 2. Formatter: Scientific notation at top-left
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0)) 
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

def _force_symmetric_ylim(ax, data):
    """
    Forces the Y-axis to be symmetric [-max, +max].
    This ensures '0' is centered and negative ticks always exist,
    guaranteeing consistent tick label widths across subplots.
    """
    if data is None or len(data) == 0: return
    
    # Calculate global max abs for this plot
    ymax = np.max(np.abs(data))
    
    # Handle flatline case
    if ymax == 0: ymax = 1.0
    
    # Add small padding (e.g. 10%)
    limit = ymax * 1.1
    
    ax.set_ylim(-limit, limit)

# =============================================================================
# HELPER: Strict Haufe Transform
# =============================================================================
def compute_haufe_patterns(S, scores):
    """
    Computes activation patterns (A) from classifier weights using the Haufe Transform.
    
    Logic: A = Cov(S, y) * Cov(y, y)^-1
    Where S are the reservoir states and y are the classifier scores.
    This enables neurophysiological interpretation of linear backward models.

    Parameters
    ----------
    S : np.ndarray
        Reservoir states of shape (n_samples, n_features).
    scores : np.ndarray
        Predicted scores/decision function values of shape (n_samples, n_classes).

    Returns
    -------
    A : np.ndarray
        Haufe patterns matrix of shape (n_features, n_classes).
    """

    S_cent = S - np.mean(S, axis=0, keepdims=True)
    y_cent = scores - np.mean(scores, axis=0, keepdims=True)
    n_samples = S.shape[0]

    cov_sy = (S_cent.T @ y_cent) / (n_samples - 1)
    cov_yy = (y_cent.T @ y_cent) / (n_samples - 1)

    try:
        if cov_yy.size == 1:
            prec_yy = 1.0 / (cov_yy + 1e-12)
        else:
            prec_yy = np.linalg.pinv(cov_yy)
    except np.linalg.LinAlgError:
        prec_yy = np.eye(cov_yy.shape[0])

    A = cov_sy @ prec_yy 
    return A

# =============================================================================
#  DYNAMICS
# =============================================================================
def analyze_dynamics(
    esn,
    classifier,
    target_time,
    state_snapshot,
    y_labels,
    times,
    n_clusters=3,
    top_n=30,
    phase_name="Unknown",
    # --- ERP Parameters ---
    erp_range=(-0.2, 0.6),
    erp_baseline_mode="mean",   
    erp_baseline_range=(-0.2, 0.0),
    # --- TFR Parameters ---
    tfr_freqs=np.arange(2, 40, 2),
    tfr_n_cycles=None,
    tfr_baseline_mode="logratio",
    tfr_baseline_range=(-0.2, 0.0),
    # --- FOOOF Parameters ---
    fooof_params=None,
    # --- Visual Parameters ---
    plot_style="poster", 
    class_names=None, 
    figsize=None,
    
    # --- NEW (workspace export only) ---
    return_results=True,               # default False: backward compatible
    export_virtual_sources=True,        # return S_virt per cluster
    export_neuron_importance=True,      # return all-unit importance
    units_timeseries_mode="top",        # "top" | "all"
    units_timeseries_indices=None,
    
    # --- NEW: inline sensor-space projection (Stage2 merged into Stage1) ---
    inline_topomaps=False,
    info=None,
    raw_X_snapshot=None,
    cov_window_half_width=0.1,
    topo_cmap="PiYG",
    topo_z_lim=3.0,
    topo_colorbar_once=True,
        
):
    """
    Primary analysis function. Performs Haufe Transform, Clustering, and Virtual Source
    reconstruction + visualization.

    Parameters
    ----------
    esn : object
        The ESN model instance (must contain `.taus` and `.n_res`).
    classifier : object
        The trained linear classifier (e.g., Ridge, LogisticRegression).
    target_time : float
        The specific time point (in seconds) to analyze feature importance (usually peak accuracy).
    state_snapshot : np.ndarray
        Reservoir states array of shape (n_epochs, n_neurons, n_times).
    y_labels : np.ndarray
        Ground truth labels for the epochs (n_epochs,).
    times : np.ndarray
        Time vector corresponding to the 3rd dimension of state_snapshot.
    n_clusters : int
        Number of clusters (virtual sources) to extract.
    top_n : int
        Number of top-performing neurons to use for clustering.
    phase_name : str
        Label for the plot title (e.g., "Test Phase").
    
    # Signal Processing Params
    erp_range : tuple
        Time limits (min, max) for plotting ERPs.
    erp_baseline_mode : str or None
        Baseline correction mode for ERPs ('mean' or None).
    tfr_freqs : np.ndarray
        Frequencies to analyze in TFR.
    tfr_baseline_mode : str
        Baseline mode for TFR (e.g., 'logratio', 'percent').
    fooof_params : dict
        Parameters for FOOOF spectral parameterization.
    
    # Projection / Topomap Params
    inline_topomaps : bool
        If True, computes sensor-space projection of virtual sources.
    info : mne.Info
        MNE Info object (required if inline_topomaps is True).
    raw_X_snapshot : np.ndarray
        Original sensor data (n_epochs, n_channels, n_times) for covariance computation.
    
    Returns
    -------
    results_out : dict
        A dictionary containing:
        - 'meta': Metadata about the analysis.
        - 'neurons': Importance scores and indices.
        - 'clusters': Dictionary of virtual sources, weights, and member indices.
        - 'figure': The matplotlib figure handle.
    """
    
    try:
        from fooof import FOOOF
        from fooof.sim.gen import gen_aperiodic
        has_fooof = True
    except ImportError:
        has_fooof = False

    if tfr_n_cycles is None: tfr_n_cycles = tfr_freqs / 2.0
    if fooof_params is None: fooof_params = {'max_n_peaks': 3, 'aperiodic_mode': 'fixed'}

    print(f"\n{'='*60}")
    print(f"Running STAGE 1 (V10 - {plot_style}) | {phase_name} | peak @ {target_time:.3f}s")

    settings = _get_plot_settings(plot_style)
    _apply_rc_params(settings)
    
    if erp_baseline_mode not in (None, "mean"):
        raise ValueError(f"erp_baseline_mode must be None or 'mean', got: {erp_baseline_mode!r}")
    
    unique_y = np.sort(np.unique(y_labels))
    n_classes = len(unique_y)
    if class_names is None:
        class_names = [f"Class {int(c)}" for c in unique_y]
    else:
        # Ensure length matches number of classes
        if len(class_names) != len(unique_y):
            raise ValueError(
                f"class_names length ({len(class_names)}) must match number of classes ({len(unique_y)})."
            )
        
    cmap = plt.get_cmap('Set1' if n_classes <= 9 else 'Set3')
    cond_colors = [cmap(i) for i in range(n_classes)]

    t_idx = int(np.argmin(np.abs(times - target_time)))
    mask_plot_erp = (times >= erp_range[0]) & (times <= erp_range[1])
    times_plot_erp = times[mask_plot_erp]
    
    mask_bl_erp = None
    if erp_baseline_mode == "mean":
        mask_bl_erp = (times >= erp_baseline_range[0]) & (times <= erp_baseline_range[1])

    # --- 1. Get Model Outputs ---
    S_t = state_snapshot[:, :, t_idx]
    if hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(S_t)
    else:
        scores = classifier.predict_proba(S_t)
    if scores.ndim == 1: scores = scores[:, np.newaxis]

    # --- 2.  Haufe Transform ---
    haufe_patterns = compute_haufe_patterns(S_t, scores)
    neuron_importance = np.max(np.abs(haufe_patterns), axis=1)
    
    top_n = min(top_n, esn.n_res)
    top_indices = np.argsort(neuron_importance)[-top_n:][::-1]
    taus = esn.taus

    # --- 3.  Clustering ---
    S_subset = state_snapshot[:, top_indices, :]
    X_features = S_subset.transpose(0, 2, 1).reshape(-1, top_n)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    cluster_labels = np.ones(top_n, dtype=int)
    Z = None
    try:
        X_for_clustering = X_scaled.T 
        Z = sch.linkage(X_for_clustering, method='ward', metric='euclidean')
        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        print(f"-> Clustering: requested {n_clusters}, got {int(np.max(cluster_labels))}")
    except Exception as e:
        print(f"-> Clustering failed: {e}")

    # --- 4. Consistent Virtual Sources ---
    virtual_sources = {}
    clusters_output = {}

    for c_id in range(1, int(np.max(cluster_labels)) + 1):
        mask = (cluster_labels == c_id)
        idxs = top_indices[mask]
        if len(idxs) == 0: continue
        
        S_virt = np.zeros((state_snapshot.shape[0], state_snapshot.shape[2]), dtype=float)
        cluster_weights = []
        
        for n_idx in idxs:
            pattern_row = haufe_patterns[n_idx]
            dom_class_idx = np.argmax(np.abs(pattern_row))
            weight = pattern_row[dom_class_idx] 
            cluster_weights.append(weight)
            S_virt += weight * state_snapshot[:, n_idx, :]
            
        S_virt /= len(idxs)
        cluster_name = f"Cluster {c_id}"
        virtual_sources[f"Cluster {c_id}"] = S_virt
        clusters_output[f"Cluster {c_id}"] = {
            'indices': idxs,
            'weights': np.array(cluster_weights),
            "S_virt": S_virt,
        }

    # --- Plotting ---
    n_sources = len(virtual_sources)
    if n_sources == 0: return {}
    n_tfr_cols = min(n_classes, 5)
    # width_ratios = [1.2] + [1] * n_tfr_cols + [1.3] + [0.9]
    width_ratios = (
        [1.2] +
        [1.0] * n_tfr_cols +
        [1.0] +      # PSD (slightly narrower)
        [0.9] +      # TOPO
        [0.08]       # COLORBAR
    )
        
    if figsize is None:
        base_w = 4 * len(width_ratios)
        base_h = 6 + 4 * n_sources
        if plot_style == "poster":
            figsize = (base_w * 1.2, base_h * 1.2)
        else:
            figsize = (base_w, base_h)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs_outer = gridspec.GridSpec(n_sources + 1, 1, height_ratios=[1.3] + [1.2] * n_sources, figure=fig)

   
    # A) Scatter: Haufe importance vs cutoff frequency fc
    # ---------------------------------------------------
    gs_map = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[0], width_ratios=[2, 1]
    )
    ax_scat = fig.add_subplot(gs_map[0])
    ax_dend = fig.add_subplot(gs_map[1])
    
    # Convert tau -> cutoff frequency: fc = 1 / (2*pi*tau)
    fc = 1.0 / (2.0 * np.pi * taus)
    
    # --- Background scatter (all units) ---
    ax_scat.scatter(
        fc,
        neuron_importance,
        c='gray',
        alpha=0.25,
        s=settings['scatter_s_bg'],
    )
    
    # Cluster colors
    cluster_colors = [ 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink' ]
    # --- Highlighted clusters ---
    for c_id in range(1, int(np.max(cluster_labels)) + 1):
        mask = (cluster_labels == c_id)
        idxs = top_indices[mask]
        c_color = cluster_colors[(c_id - 1) % len(cluster_colors)]
    
        ax_scat.scatter(
            fc[idxs],
            neuron_importance[idxs],
            s=settings['scatter_s_fg'],
            facecolors='none',
            edgecolors=c_color,
            linewidth=settings['scatter_lw'],
            label=f'Cluster {c_id}'
        )
    
    # --- X axis: log scale with linear-meaning ticks ---
    ax_scat.set_xscale('log')
    ax_scat.set_xlabel(r"$f_c$ (Hz)  [$f_c = (2\pi\tau)^{-1}$]")
    ax_scat.set_ylabel("Haufe importance")
    
    # Explicit ticks (linear values on a log axis)
    ticks_fc = np.array([2, 5, 10, 20, 30, 50, 80])
    ax_scat.set_xticks(ticks_fc)
    ax_scat.set_xticklabels([str(t) for t in ticks_fc])
    
    # Optional: limit x-range slightly beyond ticks
    ax_scat.set_xlim(ticks_fc[0] * 0.8, ticks_fc[-1] * 1.2)
    
    # --- Y axis: clean, no duplicated ticks ---
    ax_scat.minorticks_off()
    ax_scat.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_scat.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    yt = ax_scat.get_yticks()
    ax_scat.set_yticks(np.unique(np.round(yt, 6)))
    
    # Legend
    ax_scat.legend(loc='upper right', fontsize=settings['legend_size'])
    
    # Final formatting (keep your helper)
    _set_axis_format(ax_scat, 'y', style='float_1')
    
    # B) Dendrogram
    if Z is not None:
        thr = Z[-(n_clusters - 1), 2] - 1e-12 if n_clusters >= 2 else 0
        sch.dendrogram(
            Z,
            ax=ax_dend, 
            color_threshold=thr,
            above_threshold_color="k",
        )

        # ax_dend.set_title("Ward dendrogram")
        ax_dend.set_xlabel("Unit")
        ax_dend.set_ylabel("Linkage distance")
        _set_axis_format(ax_dend, 'y', style='float_1')  # optional; makes distance look cleaner
    else: ax_dend.axis('off')

    # C) Per Cluster Plots
    erp_axes_list = [] 
    tfr_cbar_drawn = False
    im_tfr_ref = None
    if inline_topomaps:
        if info is None or raw_X_snapshot is None:
            raise ValueError("inline_topomaps=True requires 'info' and 'raw_X_snapshot'.")
        t_min = target_time - cov_window_half_width
        t_max = target_time + cov_window_half_width
        win_mask = (times >= t_min) & (times <= t_max)
        
    for row_i, (src_name, S_virt) in enumerate(virtual_sources.items()):
        c_id = int(src_name.split()[-1])
        c_color = cluster_colors[(c_id - 1) % len(cluster_colors)]
        gs_row = gridspec.GridSpecFromSubplotSpec(1, len(width_ratios), subplot_spec=gs_outer[row_i + 1], width_ratios=width_ratios)
        
        # --- ERP ---
        ax_erp = fig.add_subplot(gs_row[0])
        erp_axes_list.append(ax_erp) 

        # Collect data for calculating symmetric limits
        all_erp_data = [] 
        for i, cid in enumerate(unique_y):
            m = np.mean(S_virt[y_labels == cid], axis=0)
            if erp_baseline_mode == "mean" and mask_bl_erp is not None:
                b_val = np.mean(m[mask_bl_erp])
                m = m - b_val
            
            # Slice only for plotting, but we might want limits based on visible part
            m_plot = m[mask_plot_erp]
            all_erp_data.append(m_plot)
            ax_erp.plot(times_plot_erp, m_plot, color=cond_colors[i], label=class_names[i])
            if row_i == 0:
                ax_erp.legend(
                    loc="lower right",
                    fontsize=10,
                    ncol=2,
                    columnspacing=0.8,
                    handletextpad=0.6,
                    frameon=False,
                )
        # [FIX] Force Symmetric Y-Axis limits
        _force_symmetric_ylim(ax_erp, np.concatenate(all_erp_data))
        
        ax_erp.set_title(f"{src_name}", color=c_color, fontweight='normal')
        ax_erp.axvline(0, color='k', ls='--', alpha=0.3)
        if row_i == n_clusters-1: ax_erp.set_xlabel("Time (s)")
        ax_erp.set_ylabel("Virtual source (a.u.)")
        _set_axis_format(ax_erp, 'x')
        _set_axis_format(ax_erp, 'y', style='sci')
        _set_axis_format(ax_erp, 'x')
        _set_axis_format(ax_erp, 'y', style='sci') 
      
        # --- TFR ---
        for k in range(n_tfr_cols):
            ax_tfr = fig.add_subplot(gs_row[1 + k])
            cid = unique_y[k]
            mask_c = (y_labels == cid)
            if np.sum(mask_c) == 0: continue
            
            S_in = S_virt[mask_c][:, np.newaxis, :]
            pow_raw = tfr_array_morlet(S_in, esn.fs, tfr_freqs, n_cycles=tfr_n_cycles, output='power', verbose=False)
            avg_pow = np.mean(pow_raw, axis=0)[0]
            if tfr_baseline_mode is None:
                tfr_data = avg_pow
        
            else:
                # Keep original behavior
                tfr_data = rescale(
                    avg_pow,
                    times,
                    tfr_baseline_range,
                    mode=tfr_baseline_mode,
                    verbose=False
                )
        

            ax_tfr.axvline(0, color='k', ls='--', alpha=0.3)
            im = ax_tfr.pcolormesh(
                times_plot_erp,
                tfr_freqs,
                tfr_data[:, mask_plot_erp],
                cmap='RdYlBu_r',
                shading='auto'
            )
            if (row_i == 0) and (im_tfr_ref is None):  im_tfr_ref = im
            if row_i == 0: ax_tfr.set_title(class_names[k], fontsize=settings['label_size'], pad=4)             
            if row_i == n_clusters-1:  ax_tfr.set_xlabel("Time (s)")
            if k == 0: ax_tfr.set_ylabel("Frequency (Hz)")
            else: ax_tfr.set_yticks([])
            _set_axis_format(ax_tfr, 'x')

        # --- PSD ---
        ax_psd = fig.add_subplot(gs_row[-3])
        ax_psd.set_xscale('log')
        f_axis, psd_raw = scipy.signal.welch(S_virt, fs=esn.fs, nperseg=int(esn.fs), axis=-1)
        for i, cid in enumerate(unique_y):
            m_psd = np.mean(psd_raw[y_labels == cid], axis=0)
            ax_psd.plot(f_axis, 10 * np.log10(m_psd + 1e-12), color=cond_colors[i], alpha=0.6)
            if has_fooof:
                try:
                    fm = FOOOF(**fooof_params, verbose=False)
                    fm.fit(f_axis, m_psd, [2, 40])
                    if hasattr(fm, "_ap_fit") and fm._ap_fit is not None:
                        ap_vals = fm._ap_fit
                    else:
                        ap_vals = gen_aperiodic(fm.freqs, fm.aperiodic_params_, fooof_params.get('aperiodic_mode', 'fixed'))
                    ax_psd.plot(fm.freqs, 10 * ap_vals, color=cond_colors[i], ls='--', alpha=0.9,label=class_names[i] )
                except: pass
        
        ticks = [2, 5, 10, 20, 35]
        ax_psd.set_xticks(ticks)
        ax_psd.set_xticklabels([f"{t:.0f}" for t in ticks]) 
        if row_i == n_clusters-1:ax_psd.set_xlabel("Frequency (Hz)")
 
        ax_psd.set_ylabel("Power (dB)")
        _set_axis_format(ax_psd, 'y', style='sci')
   
        if row_i == 0:
            ax_psd.legend(
                loc="lower left",
                fontsize=10,
                frameon=False,
            )
        _set_axis_format(ax_psd, 'y', style='sci')
        
        # --- TOPO (sensor-space projection) ---
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        if inline_topomaps:
            ax_topo = fig.add_subplot(gs_row[-2])
        
            # Flatten sensor data in the covariance window
            X_flat = raw_X_snapshot[:, :, win_mask].transpose(1, 0, 2).reshape(raw_X_snapshot.shape[1], -1)
        
            # Flatten the virtual source in the same window
            S_flat = S_virt[:, win_mask].reshape(-1)
        
            # Center both
            X_flat -= np.mean(X_flat, axis=1, keepdims=True)
            S_flat -= np.mean(S_flat)
        
            # Sensor-space covariance map
            cov_map = (X_flat @ S_flat) / (len(S_flat) - 1)
        
            # Per-map z-score (keeps topology visible and stable)
            m_std = np.std(cov_map)
            if m_std < 1e-15:
                m_std = 1.0
            cov_z = (cov_map - np.mean(cov_map)) / m_std
        
            im_topo, _ = mne.viz.plot_topomap(
                cov_z,
                info,
                axes=ax_topo,
                show=False,
                cmap=topo_cmap,
                vlim=(-topo_z_lim, topo_z_lim),
                contours=0
            )
        
            # Minimal title (optional)
            if row_i == 0:
                ax_topo.set_title("Topo", fontsize=settings['label_size'], pad=4)
        
            # One shared colorbar only once (avoid clutter)
            def _format_diverging_colorbar(cb, settings, nbins=5):
                """
                Enforce consistent diverging colorbar formatting:
                - symmetric ticks
                - scientific notation
                - fixed number of ticks
                """
                cb.locator = MaxNLocator(nbins=nbins)
                cb.formatter = ScalarFormatter(useMathText=True)
                cb.formatter.set_powerlimits((0, 0))  # always use scientific notation
                cb.update_ticks()
                cb.ax.tick_params(labelsize=settings['tick_size'])
    
            if topo_colorbar_once and (row_i == 0):
                cax_top = inset_axes(
                    ax_topo,
                    width="4.5%",
                    height="85%",
                    loc="lower left",
                    bbox_to_anchor=(1.04, 0.08, 1, 1),
                    bbox_transform=ax_topo.transAxes,
                    borderpad=0.0
                )
                cax_top.set_in_layout(False)
                cb_top = fig.colorbar(im_topo, cax=cax_top)
                _format_diverging_colorbar(cb_top, settings)
                cb_top.set_label("Topo (Z)", fontsize=settings['label_size'])
                ax_topo.set_aspect('equal', adjustable='box')
            # --- TFR colorbar: attach to SECOND row only ---
            if topo_colorbar_once and (row_i == 1) and (im_tfr_ref is not None):
                cax_tfr = inset_axes(
                    ax_topo,
                    width="4.5%",
                    height="85%",
                    loc="lower left",
                    bbox_to_anchor=(1.04, 0.08, 1, 1),
                    bbox_transform=ax_topo.transAxes,
                    borderpad=0.0
                )
                cax_tfr.set_in_layout(False)
                cb_tfr = fig.colorbar(im_tfr_ref, cax=cax_tfr)
                _format_diverging_colorbar(cb_tfr, settings)
                
                if isinstance(tfr_baseline_mode, str) and tfr_baseline_mode.lower() == "logratio":
                    cb_tfr.set_label("TFR (dB)", fontsize=settings['label_size'])
                else:
                    cb_tfr.set_label("TFR (a.u.)", fontsize=settings['label_size'])
             
                            
    if erp_axes_list:
        fig.align_ylabels(erp_axes_list)
    plt.show()
    
    # ---------------------------------------------------------------------
    # Workspace export 
    # ---------------------------------------------------------------------
    if return_results:
        results_out = {
            "meta": {
                "phase_name": phase_name,
                "target_time": float(target_time),
                "fs": float(esn.fs),
                "times": np.array(times, copy=True),
                "class_names": list(class_names) if class_names is not None else None,
            },
            "neurons": {
                "neuron_importance": np.array(neuron_importance, copy=True),
                "top_indices": np.array(top_indices, copy=True),
            },
            "clusters": clusters_output,  
            "figure": fig,
        }
        return results_out
    
    return clusters_output

# =============================================================================
#  GROUP-LEVEL DYNAMICS (Spatial Matching & Sign-Alignment)
# =============================================================================
def analyze_dynamics_group(
    subject_data_list,
    info,
    n_clusters=2,
    top_n=10,
    phase_name="Group Average",
    # --- ERP Parameters ---
    erp_range=(-0.2, 0.6),
    erp_baseline_mode="mean",   
    erp_baseline_range=(-0.2, 0.0),
    # --- TFR Parameters ---
    tfr_freqs=np.arange(2, 40, 2),
    tfr_n_cycles=None,
    tfr_baseline_mode="logratio",
    tfr_baseline_range=(-0.2, 0.0),
    # --- FOOOF Parameters ---
    fooof_params=None,
    # --- Visual Parameters ---
    plot_style="poster", 
    class_names=None, 
    figsize=None,
    cov_window_half_width=0.1,
    topo_cmap="PiYG",
    topo_z_lim=2.5,
    topo_colorbar_once=True,
    
    return_results=False,               
    export_virtual_sources=False
):
    """
    Group-Level Analysis Function. 
    Performs cross-subject spatial matching, sign-alignment, and constructs 
    Grand Average representations (ERP, TFR, PSD, Topo).
    """
    import matplotlib.ticker as ticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mne.baseline import rescale
    
    try:
        from fooof import FOOOF
        from fooof.sim.gen import gen_aperiodic
        has_fooof = True
    except ImportError:
        has_fooof = False
        print("Warning: FOOOF not installed. 1/f aperiodic fits will be skipped.")

    if tfr_n_cycles is None: tfr_n_cycles = tfr_freqs / 2.0
    if fooof_params is None: fooof_params = {'max_n_peaks': 4, 'aperiodic_mode': 'fixed'}

    print(f"\n{'='*60}")
    print(f"Running GROUP-LEVEL INTERPRETATION | {phase_name} | Top {top_n} Units")
    
    settings = _get_plot_settings(plot_style)
    _apply_rc_params(settings)

    classes = np.unique(subject_data_list[0]['y'])
    n_classes = len(classes)
    if class_names is None: class_names = [f"Class {int(c)}" for c in classes]
    cmap = plt.get_cmap('Set1' if n_classes <= 9 else 'Set3')
    cond_colors = [cmap(i) for i in range(n_classes)]

    # ---------------------------------------------------------------------
    # PHASE 1: Extract Spatial Topos and Align Signs
    # ---------------------------------------------------------------------
    print(f"-> Phase 1: Extracting Top {top_n} units & Spatial Sign-Alignment...")
    all_sub_data = []
    global_spatial_features = []

    for s_idx, s_data in enumerate(subject_data_list):
        X, S, y = s_data['X'], s_data['S'], s_data['y']
        times, t_peak = s_data['times'], s_data['target_time']
        clf, esn = s_data['classifier'], s_data['esn']

        t_idx = int(np.argmin(np.abs(times - t_peak)))
        S_t = S[:, :, t_idx]
        if hasattr(clf, "decision_function"): scores = clf.decision_function(S_t)
        else: scores = clf.predict_proba(S_t)
        if scores.ndim == 1: scores = scores[:, np.newaxis]

        haufe = compute_haufe_patterns(S_t, scores)
        importance = np.max(np.abs(haufe), axis=1)
        
        n_top = min(top_n, esn.n_res)
        top_indices = np.argsort(importance)[-n_top:][::-1]

        sub_sign_flips = []
        for u_idx in top_indices:
            t_min, t_max = t_peak - cov_window_half_width, t_peak + cov_window_half_width
            win_mask = (times >= t_min) & (times <= t_max)
            
            X_flat = X[:, :, win_mask].transpose(1, 0, 2).reshape(X.shape[1], -1)
            S_flat = S[:, u_idx, win_mask].reshape(-1)
            
            X_flat -= np.mean(X_flat, axis=1, keepdims=True)
            S_flat -= np.mean(S_flat)
            cov_map = (X_flat @ S_flat) / (len(S_flat) - 1)
            
            # Sign Alignment
            max_idx = np.argmax(np.abs(cov_map))
            s_flip = 1.0 if cov_map[max_idx] > 0 else -1.0
            cov_map_aligned = cov_map * s_flip
            
            cov_z = (cov_map_aligned - np.mean(cov_map_aligned)) / (np.std(cov_map_aligned) + 1e-15)
            
            global_spatial_features.append(cov_z)
            sub_sign_flips.append(s_flip)

        all_sub_data.append({
            'X': X, 'S': S, 'y': y, 'times': times, 'peak_time': t_peak, 'fs': esn.fs,
            'haufe': haufe, 'top_indices': top_indices, 'sign_flips': np.array(sub_sign_flips),
            'taus': esn.taus[top_indices], 'importance': importance[top_indices]
        })

    # ---------------------------------------------------------------------
    # PHASE 2: Global Spatial Clustering
    # ---------------------------------------------------------------------
    print(f"-> Phase 2: Global Hierarchical Clustering (N={n_clusters})...")
    X_global = np.array(global_spatial_features)
    Z = sch.linkage(X_global, method='ward', metric='euclidean')
    global_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    unit_counter = 0
    for sub in all_sub_data:
        sub['cluster_labels'] = global_labels[unit_counter : unit_counter + len(sub['top_indices'])]
        unit_counter += len(sub['top_indices'])

    # ---------------------------------------------------------------------
    # PHASE 3: Grand Average Reconstruction
    # ---------------------------------------------------------------------
    print("-> Phase 3: Reconstructing Grand Average Dynamics...")
    
    # ADDED: Include the [0.08] column for the Colorbar!
    width_ratios = [1.2] + [1.0] * n_classes + [1.0] + [0.9] + [0.08]
    
    if figsize is None:
        base_w = 4 * len(width_ratios)
        base_h = 6 + 4 * n_clusters
        figsize = (base_w * 1.2, base_h * 1.2) if plot_style == "poster" else (base_w, base_h)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs_outer = gridspec.GridSpec(n_clusters + 1, 1, height_ratios=[1.0] * (n_clusters + 1), figure=fig)
    # --- Panel A: Scatter Plot ---
    ax_scat = fig.add_subplot(gs_outer[0])
    
    all_taus = np.concatenate([d['taus'] for d in all_sub_data])
    all_imp = np.concatenate([d['importance'] for d in all_sub_data])
    fc = 1.0 / (2.0 * np.pi * all_taus)
    cluster_colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    for c_id in range(1, int(np.max(global_labels)) + 1):
        mask = (global_labels == c_id)
        if sum(mask) == 0: continue
        ax_scat.scatter(
            fc[mask], all_imp[mask], facecolors='none', 
            edgecolors=cluster_colors[(c_id-1) % len(cluster_colors)], 
            s=settings['scatter_s_fg'], lw=settings['scatter_lw'], 
            label=f'Aligned Cluster {c_id} (n={sum(mask)})'
        )

    ax_scat.set_xscale('log')
    ax_scat.set_xlabel(r"$f_c$ (Hz)")
    ax_scat.set_ylabel("Haufe importance")
    ticks_fc = [2, 5, 10, 20, 30, 50, 80]
    ax_scat.set_xticks(ticks_fc)
    ax_scat.set_xticklabels([str(t) for t in ticks_fc])
    ax_scat.legend(loc='upper right', fontsize=settings['legend_size'])
    _set_axis_format(ax_scat, 'y', style='float_1')

    # --- Panel B: Per-Cluster Reconstructions ---
    times = subject_data_list[0]['times']
    mask_plot_erp = (times >= erp_range[0]) & (times <= erp_range[1])
    times_plot = times[mask_plot_erp]
    bl_mask = (times >= tfr_baseline_range[0]) & (times <= tfr_baseline_range[1])

    erp_axes_list = []
    im_tfr_ref = None  # Store for Colorbar
    clusters_output = {}
    for row_i, c_id in enumerate(range(1, int(np.max(global_labels)) + 1)):
        c_color = cluster_colors[(c_id-1) % len(cluster_colors)]
        gs_row = gridspec.GridSpecFromSubplotSpec(1, len(width_ratios), subplot_spec=gs_outer[row_i + 1], width_ratios=width_ratios)
        
        ga_erps, ga_tfrs, ga_psds = {c: [] for c in classes}, {c: [] for c in classes}, {c: [] for c in classes}
        ga_topos = []
        
        sub_virtual_sources = []
        sub_indices_list = []
        sub_sign_flips_list = []
        for sub in all_sub_data:
            mask = (sub['cluster_labels'] == c_id)
            if not np.any(mask): 
                if export_virtual_sources:
                    sub_virtual_sources.append(None)
                    sub_indices_list.append(None)
                    sub_sign_flips_list.append(None)
                continue
            
            idxs, s_flips = sub['top_indices'][mask], sub['sign_flips'][mask]
            S_virt_sub = np.zeros((sub['S'].shape[0], sub['S'].shape[2]))
            
            for n_idx, s_flip in zip(idxs, s_flips):
                w = sub['haufe'][n_idx, np.argmax(np.abs(sub['haufe'][n_idx]))]
                S_virt_sub += w * s_flip * sub['S'][:, n_idx, :]
            S_virt_sub /= len(idxs)
            
            if export_virtual_sources:
                sub_virtual_sources.append(S_virt_sub)
                sub_indices_list.append(idxs)
                sub_sign_flips_list.append(s_flips)
            
            # Sub-level ERP
            for c in classes:
                m_erp = np.mean(S_virt_sub[sub['y'] == c], axis=0)
                if erp_baseline_mode == "mean":
                    m_erp -= np.mean(m_erp[(times >= erp_baseline_range[0]) & (times <= erp_baseline_range[1])])
                ga_erps[c].append(m_erp[mask_plot_erp])
            
            # Sub-level PSD
            f_psd, psd_raw = scipy.signal.welch(S_virt_sub, fs=sub['fs'], nperseg=int(sub['fs']), axis=-1)
            for c in classes:
                ga_psds[c].append(np.mean(psd_raw[sub['y'] == c], axis=0))
            
            # Sub-level TFR
            for c in classes:
                pow_raw = tfr_array_morlet(S_virt_sub[sub['y'] == c][:, np.newaxis, :], sub['fs'], tfr_freqs, n_cycles=tfr_n_cycles, output='power', verbose=False)
                avg_pow = np.mean(pow_raw, axis=0)[0]
                
                if tfr_baseline_mode == "logratio":
                    b_val = np.mean(avg_pow[:, bl_mask], axis=1, keepdims=True)
                    tfr_res = 10 * np.log10(avg_pow / (b_val + 1e-12))
                else:
                    tfr_res = rescale(avg_pow, times, tfr_baseline_range, mode=tfr_baseline_mode, verbose=False)
                ga_tfrs[c].append(tfr_res[:, mask_plot_erp])
            
            # Sub-level Topo
            t_min, t_max = sub['peak_time'] - cov_window_half_width, sub['peak_time'] + cov_window_half_width
            win_mask = (times >= t_min) & (times <= t_max)
            X_f = sub['X'][:, :, win_mask].transpose(1, 0, 2).reshape(sub['X'].shape[1], -1)
            S_f = S_virt_sub[:, win_mask].reshape(-1)
            X_f -= np.mean(X_f, axis=1, keepdims=True)
            cov_map = (X_f @ (S_f - np.mean(S_f))) / (len(S_f) - 1)
            ga_topos.append(cov_map)

        if len(ga_topos) == 0: continue

        
        clusters_output[f"Cluster {c_id}"] = {
            "subject_virtual_sources": sub_virtual_sources if export_virtual_sources else None,
            "subject_unit_indices": sub_indices_list if export_virtual_sources else None,
            "subject_sign_flips": sub_sign_flips_list if export_virtual_sources else None,
            "ga_erps": ga_erps,     
            "ga_tfrs": ga_tfrs,     
            "ga_psds": ga_psds,
            "ga_topos_raw": ga_topos 
        }

        # --- 1. Plot Grand Average ERP ---
        ax_erp = fig.add_subplot(gs_row[0])
        erp_axes_list.append(ax_erp)
        for i, c in enumerate(classes):
            ax_erp.plot(times_plot, np.mean(ga_erps[c], axis=0), color=cond_colors[i], label=class_names[i])
        ax_erp.set_title(f"Cluster {c_id}", color=c_color)
        ax_erp.axvline(0, color='k', ls='--', alpha=0.3)
        if row_i == n_clusters-1: ax_erp.set_xlabel("Time (s)")
        ax_erp.set_ylabel("Virtual source (a.u.)")
        if row_i == 0: ax_erp.legend(loc="lower right", fontsize=10, frameon=False, ncol=2)
        _force_symmetric_ylim(ax_erp, np.concatenate([np.mean(ga_erps[c], axis=0) for c in classes]))
        _set_axis_format(ax_erp, 'x')
        _set_axis_format(ax_erp, 'y', style='sci')

        # --- 2. Plot Grand Average TFR ---
        for k, c in enumerate(classes):
            ax_tfr = fig.add_subplot(gs_row[1 + k])
            m_tfr = np.mean(ga_tfrs[c], axis=0)
            im = ax_tfr.pcolormesh(times_plot, tfr_freqs, m_tfr, cmap='RdYlBu_r', shading='auto')
            
            if (row_i == 0) and (im_tfr_ref is None): im_tfr_ref = im  
                
            ax_tfr.axvline(0, color='k', ls='--', alpha=0.3)
            if row_i == 0: ax_tfr.set_title(class_names[k], fontsize=settings['label_size'])
            if k == 0: ax_tfr.set_ylabel("Freq (Hz)")
            else: ax_tfr.set_yticks([])
            if row_i == n_clusters-1: ax_tfr.set_xlabel("Time (s)")
            _set_axis_format(ax_tfr, 'x')

        # --- 3. Plot Grand Average PSD ---
        ax_psd = fig.add_subplot(gs_row[-3])
        for i, c in enumerate(classes):
            m_psd = np.mean(ga_psds[c], axis=0)
            ax_psd.plot(f_psd, 10 * np.log10(m_psd + 1e-12), color=cond_colors[i], alpha=0.6)
            
            if has_fooof:
                try:
                    fm = FOOOF(**fooof_params, verbose=False)
                    fm.fit(f_psd, m_psd, [2, 40])
                    if hasattr(fm, "_ap_fit") and fm._ap_fit is not None:
                        ap_vals = fm._ap_fit
                    else:
                        ap_vals = gen_aperiodic(fm.freqs, fm.aperiodic_params_, fooof_params.get('aperiodic_mode', 'fixed'))
                    ax_psd.plot(fm.freqs, 10 * ap_vals, color=cond_colors[i], ls='--', alpha=0.9)
                except Exception as e:
                    pass
                
        ax_psd.set_xscale('log')
        ticks = [2, 5, 10, 20, 35]
        ax_psd.set_xticks(ticks)
        ax_psd.set_xticklabels([f"{t:.0f}" for t in ticks])
        if row_i == n_clusters-1: ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("Power (dB)")
        _set_axis_format(ax_psd, 'y', style='sci')

        # --- 4. Plot Grand Average Topo ---
        ax_topo = fig.add_subplot(gs_row[-2])
        m_cov = np.mean(ga_topos, axis=0)
        m_topo_z = (m_cov - np.mean(m_cov)) / (np.std(m_cov) + 1e-15)
        
        im_topo, _ = mne.viz.plot_topomap(m_topo_z, info, axes=ax_topo, show=False, cmap=topo_cmap, vlim=(-topo_z_lim, topo_z_lim), contours=0)
        if row_i == 0: ax_topo.set_title("Topo", fontsize=settings['label_size'], pad=4)

        # --- 5. COLORBAR LOGIC ---
        def _format_diverging_colorbar(cb, settings, nbins=5):
            cb.locator = MaxNLocator(nbins=nbins)
            cb.formatter = ticker.ScalarFormatter(useMathText=True)
            cb.formatter.set_powerlimits((0, 0))
            cb.update_ticks()
            cb.ax.tick_params(labelsize=settings['tick_size'])

        if topo_colorbar_once and (row_i == 0):
            cax_top = inset_axes(
                ax_topo, width="4.5%", height="85%", loc="lower left",
                bbox_to_anchor=(1.04, 0.08, 1, 1), bbox_transform=ax_topo.transAxes, borderpad=0.0
            )
            cax_top.set_in_layout(False)
            cb_top = fig.colorbar(im_topo, cax=cax_top)
            _format_diverging_colorbar(cb_top, settings)
            cb_top.set_label("Topo (Z)", fontsize=settings['label_size'])
            ax_topo.set_aspect('equal', adjustable='box')
            
        if topo_colorbar_once and (row_i == 1) and (im_tfr_ref is not None):
            cax_tfr = inset_axes(
                ax_topo, width="4.5%", height="85%", loc="lower left",
                bbox_to_anchor=(1.04, 0.08, 1, 1), bbox_transform=ax_topo.transAxes, borderpad=0.0
            )
            cax_tfr.set_in_layout(False)
            cb_tfr = fig.colorbar(im_tfr_ref, cax=cax_tfr)
            _format_diverging_colorbar(cb_tfr, settings)
            
            if isinstance(tfr_baseline_mode, str) and tfr_baseline_mode.lower() == "logratio":
                cb_tfr.set_label("TFR (dB)", fontsize=settings['label_size'])
            else:
                cb_tfr.set_label("TFR (a.u.)", fontsize=settings['label_size'])

    if erp_axes_list: fig.align_ylabels(erp_axes_list)
    plt.show()

    # ---------------------------------------------------------------------
    # NEW: Workspace export 
    # ---------------------------------------------------------------------
    if return_results:
        results_out = {
            "meta": {
                "phase_name": phase_name,
                "times": np.array(times, copy=True),
                "classes": classes,
                "class_names": class_names,
                "n_clusters": n_clusters,
                "top_n": top_n
            },
            "clusters": clusters_output,  
            "figure": fig,
        }
        return results_out
    
    return clusters_output
