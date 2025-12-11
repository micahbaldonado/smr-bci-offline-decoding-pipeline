# Motor Imagery Classification from EEG Signals

### Offline Decoding Pipeline for SMR-Based Brain–Computer Interfaces

This repository contains the complete offline analysis pipeline developed for our SMR motor-imagery EEG study. The implementation mirrors the methodology described in the written report, integrating disciplined preprocessing, physiologically grounded feature extraction, and systematic benchmarking across multiple decoding models. The codebase recreates each stage of the analysis exactly as performed in the study and enables full replication of the results.

---

## Overview

The project focuses on decoding sensorimotor rhythms (SMRs) recorded during left–right and up–down motor imagery. Each component follows well-established principles in EEG-based BCI research: noise-aware preprocessing, representation learning aligned with SMR neurophysiology, and rigorous model evaluation. The pipeline processes raw EEG from 27 gelled electrodes, extracts discriminative spatial patterns, and benchmarks a range of classifiers including linear, nonlinear, ensemble, and deep neural architectures.

All offline accuracy is evaluated using **Percent Valid Correct (PVC)**, matching the classification component of the online decoder and aligning with the framework discussed in the report.

**Domain Expertise**: Direct participation in the experimental data collection phase provided critical insight into signal quality variations, artifact characteristics, and trial-level quality assessment. This first-hand understanding of data generation processes informed principled preprocessing design decisions, particularly in artifact handling and trial filtering strategies.

---

## Pipeline Summary

### **1. Preprocessing**

The preprocessing pipeline replicates all major steps described in the report:

- Selection of 27 gelled electrodes over the sensorimotor cortex  
- 8–30 Hz Butterworth band-pass filtering  
- Common Average Referencing  
- Extraction of fixed 3000-sample epochs centered on ball onset  
- Downsampling from 1000 Hz to 200 Hz  
- Exclusion of all aborted trials to maintain consistency with PVC  

These decisions produce consistent, information-rich epochs optimized for SMR decoding and align with the structure used in the written analysis.

---

### **2. Feature Extraction: Common Spatial Patterns (CSP)**

The repository includes a full CSP implementation, featuring:

- Regularized covariance estimation  
- Six CSP pairs (12 log-variance features per trial)  
- Fold-specific CSP recomputation inside cross-validation to prevent leakage  
- Tools for generating CSP topographic maps  

These spatial filters consistently localize to the expected sensorimotor regions and provide a strong representational basis for several of the models.

---

### **3. Classifier Suite**

All classifiers evaluated in the report are implemented here:

- **CSP + LDA**  
- **CSP + MLP**  
- **Random Forest**  
- **XGBoost**  
- **EEGNet** (temporal convolution, depthwise spatial filtering, separable convolution, batch normalization, dropout)

EEGNet is implemented directly from the 2018 architecture and trained on raw 27-channel EEG epochs. All classifiers use stratified 5-fold cross-validation with fixed random seeds for deterministic replication.

---

## Evaluation and Results

The offline models are evaluated across all four subject-task conditions (S1-LR, S1-UD, S2-LR, S2-UD) and compared against the online PVC baseline recorded during the experiment. Every offline model exceeded the online decoder's accuracy.

Highlights from the benchmarking:

- CSP+LDA establishes a strong linear baseline  
- CSP+MLP captures additional nonlinear structure  
- EEGNet learns spatiotemporal patterns directly from raw EEG  
- XGBoost achieves the strongest overall improvement across conditions  

These findings reflect the broader trend observed in the report: disciplined preprocessing and well-structured representations often determine model success more than architectural complexity.

Below are the evaluation results:
<img width="2024" height="1091" alt="image" src="https://github.com/user-attachments/assets/a8fa5529-4bb9-494f-b821-5e4fb46b29bb" />

---

## Repository Structure

```
├── CSP.py                          # Common Spatial Patterns implementation
├── LDA.py                          # Linear Discriminant Analysis baseline
├── final_pipeline.py               # Full offline pipeline script
├── final_smr_eeg_bci_pipeline.ipynb # Complete Jupyter notebook version
├── requirements.txt                # Python dependencies
│
├── CSP_outputs/                   # CSP spatial maps and discriminative patterns
│   ├── csp_topomaps.png
│   ├── S1_LR.png, S1_UD.png
│   └── S2_LR.png, S2_UD.png
│
├── eegnet_training_plots/          # Training and validation curves
│   └── [experiment]_training_curves.png
│
├── report_figures/                 # Comparison plots used in the report
│   ├── pvc_vs_all_models.png
│   ├── combined_radar_plot.png
│   ├── model_radar_plots.png
│   ├── PVC_plot.png
│   └── trial_outcomes_distribution.png
│
└── ORIGINAL_DATA_Group1_BCI_SMR/  # Raw EEG (.mat) files from the experiment
    ├── Session1/
    └── Session2/
```

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run via notebook

```bash
jupyter notebook final_smr_eeg_bci_pipeline.ipynb
```

### Run via script

```bash
python final_pipeline.py
```

The first run caches MATLAB files for faster subsequent execution.

---

## Acknowledgments

This project was conducted collaboratively as part of our group's SMR motor-imagery study. 

Joseph, Dheemant, Defne, JD, and Alexis contributed extensively to data collection, model design, and the analytical direction of the project, and their work directly shaped the methods reproduced in this repository.

---

## Citation

If referencing this repository in academic work, please cite the corresponding course report:

Baldonado, M. (2025). Decoding Sensorimotor Rhythms: Offline Classification of Motor Imagery for EEG-Based Brain–Computer Interfaces. Carnegie Mellon University.

---

## References

This project implements standard BCI methodologies:

- **CSP**: Blankertz et al. (2008) - "The BCI Competition III"
- **EEGNet**: Lawhern et al. (2018) - "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
- **Motor Imagery**: Pfurtscheller & Neuper (2001) - "Motor imagery and direct brain-computer communication"

---

## Author

**Micah Baldonado**  
Carnegie Mellon University  
Fall 2025
