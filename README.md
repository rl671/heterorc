# HeteroRC: Interpretable Heterogeneous Reservoir Computing for Neural Time-Series Decoding

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to the official repository for **HeteroRC**, a biologically inspired and interpretable decoding framework for dynamic neural responses (EEG/MEG/LFPs).

HeteroRC is designed to overcome the core limitations of conventional decoding pipelines in cognitive neuroscience, offering three major advantages :

1. ⚡ **Direct Decoding on Raw Time-Series:** It operates strictly on raw, multichannel time-series data. Absolutely no prior transformation, frequency filtering, or manual feature engineering is required. 
2. 🧠 **Sensitivity to Diverse Neural Dynamics:** Traditional linear decoders (e.g., LDA, SVM) applied to raw amplitudes are primarily sensitive to phase-locked *evoked potentials*. In contrast, HeteroRC can robustly and simultaneously decode information embedded in **induced oscillatory power, inter-site phase synchronization (ISPC), and aperiodic spectral modulations (slope/intercept)** .
3. 🪶 **Lightweight & Data-Efficient:** Unlike heavy CNN- or RNN-based deep learning algorithms, HeteroRC requires no gradient-based backpropagation through time. It runs effortlessly on a standard laptop CPU. This extreme efficiency makes it perfectly suited for cognitive neuroscience studies characterized by **small sample sizes and limited trial counts** .

---

## 🌟 Key Features

* **Time-Series Decoding:** Performs robust time-resolved cross-validation decoding directly on raw multichannel data, accurately pinpointing when task-relevant information emerges .
* **Cross-Temporal Generalization:** Computes Temporal Generalization Matrices (TGMs) by training and testing across all time points, revealing whether neural representations are dynamically evolving or temporally stable .
* **Cross-Task/Condition/Session Generalization:** Provides dedicated pipelines for rigorous independent train/test evaluations. Seamlessly supports scenarios where models are trained on one session, task, or condition and evaluated on a completely independent dataset .
* **Individual-Level Interpretability:** Avoids the "black box" problem. By converting uninterpretable classifier weights into activation patterns via the Haufe Transform, it automatically extracts latent virtual source signals and evaluates them across temporal (ERPs), spectral (TFRs, FOOOF-parameterized PSDs), and spatial (Topomaps) domains .
* **Group-Level Inference:** Features a novel sensor-space matching and sign-alignment algorithm. It projects idiosyncratic reservoir latent activities back to a common physical EEG sensor space, enabling global spatial clustering and the reconstruction of Grand-Average neural motifs across multiple participants .

---

## 📂 Repository Structure

* **`heterorc.py`**: The core HeteroRC module, acting as a fixed recurrent feature extractor with multiscale time constants, alongside robust cross-validated decoding functions.
* **`heterorc_interpretation.py`**: The comprehensive interpretation suite for both individual-level and group-level virtual source reconstruction and multi-domain visualization.
* **`simulate_eeg.py`**: Built-in tools for generating controlled synthetic EEG datasets (evoked, induced, ISPC, aperiodic) to validate decoding models.
* **`Tutorial*_*.ipynb`**: Interactive Jupyter Notebooks to help you get started quickly (see *Getting Started* below).
* **`paper_scripts/`**: Contains the exact scripts and analytical pipelines used to generate the figures and results presented in our paper. 
* **`BCI2a/`**: Data processing scripts specifically tailored for the Graz Motor Imagery dataset (BCI Competition IV-2a).

---

## ⚙️ Installation & Dependencies

Clone the repository to your local machine:
```bash
git clone https://github.com/rl671/heterorc.git
cd heterorc
```

**Core Dependencies:**
* `numpy`
* `scipy`
* `scikit-learn`
* `matplotlib`
* `mne` (for Morlet wavelets and topomap projections) * `fooof` (optional, but recommended for aperiodic spectral parameterization) ## 🗂️ Repository Structure

## 🚀 Getting Started (Tutorials)

We provide three interactive Jupyter Notebooks to help you integrate HeteroRC into your own analytical pipelines:

1. **`Tutorial1_decoding_with_HeteroRC.ipynb`**: Start here. Uses `simulate_eeg.py` to generate controlled neural dynamics. Demonstrates why standard linear decoders fail on non-phase-locked signals and how to configure HeteroRC for robust time-resolved and cross-temporal generalization decoding.
2. **`Tutorial2_individual_level_interpretation.ipynb`**: Applies HeteroRC to real EEG data (Motor Imagery). Demonstrates strict cross-validation for peak selection and uses `analyze_dynamics` to extract idiosyncratic virtual sources, ERPs, TFRs, and Topomaps.
3. **`Tutorial3_group_level_interpretation.ipynb`**: Demonstrates the `analyze_dynamics_group` function. Shows how to project individual reservoir nodes to a common sensor space, perform spatial clustering, and extract Grand-Average neural dynamics for cross-subject statistical testing.

## 📖 Citation

If you use HeteroRC or the associated interpretability framework in your research, please cite our paper:

```bibtex
@article{Lu2026HeteroRC,
  title={HeteroRC: Decoding latent information from dynamic neural responses with interpretable heterogeneous reservoir computing},
  author={Lu, Runhao and Liu, Sichao and Liu, Yanan and Duncan, John and Henson, Richard N. and Woolgar, Alexandra},
  journal={bioRxiv},
  year={2026}
}
```
