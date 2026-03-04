# HeteroRC: Heterogeneous Reservoir Computing for EEG/MEG Decoding & Interpretation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper: **"HeteroRC: Decoding latent information from dynamic neural responses with interpretable heterogeneous reservoir computing"**

HeteroRC is a neurobiologically-inspired, computationally efficient machine learning framework designed for the time-resolved decoding and physiological interpretation of complex macroscopic neural dynamics (EEG/MEG). 

Unlike standard linear decoders that require extensive pre-filtering or rigid physiological assumptions, HeteroRC leverages a high-dimensional, temporal reservoir with a log-normal distribution of intrinsic time constants to inherently disentangle spectrotemporal patterns (e.g., ERPs, induced oscillations, and 1/f aperiodic activity).

## 🌟 Key Features
- **Time-Resolved Decoding**: Captures rich temporal dynamics without sliding windows.
- **Strict Methodological Isolation**: Zero data-leakage global scaling and nested cross-validation protocols.
- **Individual-Level Interpretation**: Reconstructs latent virtual source signals (ERPs, TFRs, FOOOF) using Haufe's transform and Ward clustering.
- **Group-Level Generalization (Sensor-Space Matching)**: Maps idiosyncratic latent reservoir units to a common physical sensor space, aligning spatial polarities to reconstruct grand-average neural motifs across participants.

---

## 📁 Repository Structure

The repository is logically divided into core functional modules and experiment scripts used to reproduce the figures in the manuscript.

### `src/` - Core Framework
- `heterorc.py`: The core Heterogeneous Reservoir Computing network, including modules for strictly isolated cross-validation and temporal smoothing.
- `heterorc_interpretation.py`: The two-level interpretation framework (Individual & Group-level), containing Haufe transforms, dynamic clustering, and multi-domain visualizations (ERPs, TFRs, PSDs, Topomaps).
- `simulate_eeg.py`: A robust EEG simulator capable of generating ERPs, induced power modulations, inter-site phase clustering (ISPC), and 1/f aperiodic shifts.

### `scripts/` - Reproducing Paper Figures
- **Figure 2**: Simulation benchmarking against LDA/SVM (`Fig2_RC_time_decoding.py`, `Fig2_Simulation_univariate.py`).
- **Figure 3**: Cross-temporal generalization (CTG) and individual interpretation on the **BCI Competition IV-2a** dataset.
- **Figure 4**: Time-resolved decoding and individual latent dynamics on the **Attentional Priority** dataset.
- **Figure 5**: Group-level interpretation via sensor-space matching for both BCI and Attentional Priority datasets.
- **Figure S2**: Performance comparison against Wavelet and Filter-Hilbert transformations (`FigS2_Wavelet_cmp_simu.py`).


