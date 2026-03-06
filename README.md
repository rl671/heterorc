# HeteroRC: Interpretable Heterogeneous Reservoir Computing for Neural Decoding

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HeteroRC** is a biologically inspired and interpretable decoding framework designed to extract latent information from dynamic neural responses (e.g., EEG, MEG, LFPs). 

Conventional linear decoders operating on instantaneous amplitude signals effectively capture phase-locked, evoked responses, but they often fail to recover information embedded in nonlinear, non-phase-locked dynamics. **HeteroRC** overcomes this by projecting raw neural time series into a high-dimensional recurrent state space with diverse intrinsic time constants, mimicking the ability of neural populations to integrate multiscale dynamics. 

Crucially, HeteroRC operates directly on raw neural time series without requiring explicit feature engineering, and it includes a robust interpretability framework to map latent reservoir dynamics back to physiological virtual sources and sensor-space topographies.

---

## 🌟 Key Features

* **Multiscale Temporal Integration**: Reservoir units are endowed with heterogeneous intrinsic time constants sampled from a log-normal distribution, enabling simultaneous capture of fast transient responses and slow persistent dynamics.
* **Raw Data Processing**: Decodes evoked responses, induced oscillatory power, phase synchrony, and aperiodic modulations directly from raw multichannel time series without manual feature engineering.
* **White-Box Interpretability**: Utilizes the Haufe transform and hierarchical clustering to open the recurrent "black box," extracting interpretable temporal (ERP), spectral (TFR/PSD), and spatial (Topomap) motifs.
* **Group-Level Inference**: Implements a novel *Sensor-Space Matching* and *Sign-Alignment* approach to identify consistent, population-level neural motifs across idiosyncratic, randomly initialized individual reservoirs.

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
