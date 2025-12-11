# Motor Imagery Classification from EEG Signals: A Comparative Study of Feature Extraction and Classification Methods

## Technical Expertise & Research Contributions

This project demonstrates end-to-end development of a **production-grade machine learning pipeline** for motor imagery classification from electroencephalography (EEG) signals. The work encompasses signal processing, feature engineering, model development, and rigorous evaluation across multiple classification paradigms. Direct participation in the data collection phase provided domain expertise that informed preprocessing design decisions and artifact handling strategies, demonstrating the integration of domain knowledge with machine learning methodology.

**Core Technical Competencies:**

- **Signal Processing & Preprocessing**: Designed and implemented a comprehensive EEG preprocessing pipeline incorporating Common Average Referencing (CAR), multi-stage bandpass filtering (8-30 Hz Butterworth, 4th order), and zero-phase decimation. Direct involvement in data collection provided critical insight into noise characteristics, artifact patterns, and signal quality variations, enabling data-driven preprocessing parameter selection.

- **Feature Engineering**: Developed a complete **Common Spatial Patterns (CSP)** implementation from first principles, including generalized eigendecomposition with regularization, optimal component selection via eigenvalue analysis, and log-power feature extraction. The implementation handles channel ordering, spatial pattern visualization, and extends to Filter Bank CSP for multi-band analysis.

- **Model Development & Evaluation**: Systematically implemented and evaluated **5 distinct classification approaches** spanning linear models (LDA), neural networks (MLP, EEGNet), and ensemble methods (Random Forest, XGBoost). Employed stratified 5-fold cross-validation with rigorous reproducibility controls to ensure statistically sound performance comparisons.

- **Deep Learning Architecture**: Implemented **EEGNet**, a state-of-the-art convolutional architecture for EEG classification, including depthwise separable convolutions, batch normalization, and dropout regularization. Trained models with early stopping and hyperparameter optimization for optimal generalization.

- **Domain Knowledge Integration**: Leveraged understanding of sensorimotor rhythm neurophysiology and motor imagery paradigms to inform feature selection, trial filtering criteria, and model interpretation. First-hand experience with data collection challenges enabled principled handling of trial quality assessment and artifact rejection.

- **Research Methodology**: Established robust evaluation frameworks with domain-specific metrics (PVC, PTC), comprehensive visualization suites (topomaps, training curves, performance comparisons), and reproducible experimental protocols.

---

## Project Overview

This project presents a **comprehensive offline analysis framework** for motor imagery classification from electroencephalography (EEG) signals. The system implements a complete machine learning pipeline from raw signal preprocessing through model evaluation, demonstrating systematic improvements over baseline online performance across multiple experimental conditions.

**Research Objectives**: The work addresses a fundamental challenge in brain-computer interface (BCI) systems: translating noisy, high-dimensional EEG signals into reliable motor imagery classifications. By implementing and comparing multiple feature extraction and classification paradigms, the project provides empirical evidence for optimal model selection strategies in motor imagery BCI applications.

**Methodological Contribution**: The research demonstrates systematic evaluation of traditional feature engineering approaches (CSP) versus end-to-end deep learning methods (EEGNet), providing insights into the trade-offs between interpretability and performance in EEG classification tasks.

**Domain Knowledge Integration**: Direct participation in the experimental data collection phase provided critical domain knowledge regarding signal quality variations, artifact characteristics, and trial-level quality assessment. This first-hand understanding of data generation processes informed principled preprocessing design decisions, particularly in artifact handling and trial filtering strategies, demonstrating the value of domain expertise in machine learning pipeline development.

### Research Contributions

- **Novel Feature Extraction Pipeline**: Implemented Common Spatial Patterns (CSP) from first principles with optimized regularization and component selection, achieving robust spatial feature representations for motor imagery classification

- **Comprehensive Model Benchmarking**: Developed and evaluated 5 distinct classification paradigms (LDA, MLP, EEGNet, Random Forest, XGBoost) with rigorous cross-validation, establishing empirical performance baselines across multiple experimental conditions

- **Rigorous Experimental Design**: Implemented stratified 5-fold cross-validation with comprehensive reproducibility controls, ensuring statistically sound performance comparisons and generalizable results

- **Advanced Visualization Framework**: Created publication-quality visualizations including topographic maps, training dynamics, multi-dimensional performance comparisons, and model interpretability analyses

- **Production-Grade Implementation**: Developed modular, maintainable codebase with efficient data caching, comprehensive error handling, and extensive documentation, demonstrating software engineering best practices in research contexts  

---

## 🏗️ Project Structure

```
├── CSP.py                          # Common Spatial Patterns implementation
├── LDA.py                          # Linear Discriminant Analysis classifier
├── final_pipeline.py              # Main analysis pipeline (Jupyter notebook)
├── final_smr_eeg_bci_pipeline.ipynb  # Complete notebook version
├── requirements.txt               # Python dependencies
│
├── CSP_outputs/                   # CSP visualizations and topomaps
│   ├── csp_topomaps.png
│   ├── S1_LR.png, S1_UD.png
│   └── S2_LR.png, S2_UD.png
│
├── eegnet_training_plots/         # EEGNet training curves
│   └── [experiment]_training_curves.png
│
├── report_figures/                # Publication-quality figures
│   ├── pvc_vs_all_models.png      # Model comparison
│   ├── combined_radar_plot.png    # Radar plot visualization
│   ├── model_radar_plots.png      # Individual model radars
│   ├── PVC_plot.png               # Online performance baseline
│   └── trial_outcomes_distribution.png
│
└── ORIGINAL_DATA_Group1_BCI_SMR/  # Raw EEG data (MATLAB format)
    ├── Session1/
    └── Session2/
```

---

## Methodology

### 1. Signal Preprocessing & Feature Engineering

The preprocessing pipeline implements a multi-stage transformation from raw EEG signals to analysis-ready feature representations. Design decisions were informed by domain knowledge of EEG signal characteristics and empirical analysis of data quality metrics.

**Pipeline Components:**
- **Channel Selection**: Extracts 27 gelled channels from 68 total electrodes based on signal quality assessment and experimental protocol specifications
- **Common Average Referencing (CAR)**: Removes common-mode noise and reference electrode artifacts through spatial filtering, improving signal-to-noise ratio
- **Spectral Filtering**: 8-30 Hz Butterworth bandpass filter (4th order) targeting sensorimotor rhythm (SMR) bands (mu: 8-13 Hz, beta: 13-30 Hz) associated with motor imagery
- **Trial Segmentation**: 3000-sample temporal windows (1.5s pre-stimulus + 1.5s post-stimulus) optimized for capturing motor imagery onset dynamics
- **Downsampling**: Zero-phase decimation (factor of 5: 1000 Hz → 200 Hz) preserving SMR frequency content while reducing computational complexity
- **Quality Control**: Systematic exclusion of trials < 1.5s duration and aborted trials based on empirical quality metrics and domain knowledge of motor imagery temporal requirements

### 2. Common Spatial Patterns (CSP) Feature Extraction

**Mathematical Foundation:**
The CSP algorithm solves the generalized eigenvalue problem: `C₁v = λ(C₁ + C₂)v`, where `C₁` and `C₂` are class-conditional covariance matrices. This identifies spatial filters that maximize variance for one class while minimizing variance for the other.

**Implementation Details:**
- Regularized covariance estimation with ridge parameter (λ = 1e-6) to ensure numerical stability and positive definiteness
- Optimal component selection via eigenvalue analysis, selecting `m_pairs` components from both extremes of the eigenvalue spectrum (6 pairs = 12 features)
- Log-power feature extraction: `log(var(Z))` where `Z = W^T X` for each CSP component, providing discriminative power-based features
- Spatial pattern visualization using MNE-Python for interpretable topographic representations of learned filters

**Technical Contribution**: Robust implementation handling channel ordering, electrode montage mapping, and spatial pattern interpretation for physiologically meaningful feature extraction.

### 3. Classification Models

#### **CSP + Linear Discriminant Analysis (Baseline)**
- Linear classifier operating on CSP-derived log-power features
- Shared covariance assumption (pooled covariance matrix)
- Provides interpretable baseline with fast inference
- Regularization parameter: λ = 1e-4

#### **CSP + Multi-Layer Perceptron**
- Architecture: `12 (input) → 64 (hidden) → 32 (hidden) → 2 (output)`
- Activation: ReLU for hidden layers
- Optimization: Adam with learning rate 1e-3
- Training: 200 epochs with batch processing
- Regularization: Implicit via architecture and early stopping

#### **EEGNet (Deep Convolutional Architecture)**
- Architecture: Depthwise separable convolutions optimized for EEG signals
- Components: Temporal convolution (64-tap), spatial depthwise convolution, separable convolution, batch normalization, dropout (p=0.25)
- Training: 800 epochs with early stopping, learning rate 2e-4, weight decay 1e-3
- **Key Innovation**: Processes raw EEG directly without CSP preprocessing, learning spatial-temporal features end-to-end

#### **Random Forest Ensemble**
- Configuration: 200 decision trees, `sqrt` max features per split
- Training: Bootstrap aggregation on CSP features
- Hyperparameters: Optimized for high-dimensional feature spaces

#### **XGBoost Gradient Boosting**
- Configuration: 800 estimators, max_depth=5, learning_rate=0.08
- Regularization: L1 (α=0.05), L2 (λ=0.5), gamma=0.05
- Training: Early stopping with 30-round patience, subsample=0.9, colsample_bytree=0.9
- **Optimization**: Hyperparameters tuned specifically for BCI classification tasks

### 4. Experimental Design & Evaluation

**Cross-Validation Strategy:**
- **5-fold Stratified Cross-Validation**: Ensures balanced class distribution across folds, maintaining representative train/test splits
- **Reproducibility Framework**: Fixed random seeds (42, 123) across all experiments, ensuring deterministic results and enabling exact replication

**Performance Metrics:**
- **PVC** (Percent Valid Correct): Online performance metric excluding aborted trials, providing conservative baseline for comparison
- **PTC** (Percent Trial Correct): Online performance including all trials, representing real-world system performance
- **Cross-Validation Accuracy**: Mean and standard deviation across folds for offline models, enabling statistical comparison

**Statistical Rigor**: All models evaluated under identical conditions with consistent preprocessing, feature extraction, and evaluation protocols, ensuring fair comparison and generalizable conclusions.

---

## Results & Analysis

### Model Performance Evaluation

Comprehensive evaluation across **4 experimental conditions** representing different subject-task combinations:
- Subject 1: Left-Right (LR) motor imagery classification
- Subject 1: Up-Down (UD) motor imagery classification  
- Subject 2: Left-Right (LR) motor imagery classification
- Subject 2: Up-Down (UD) motor imagery classification

**Empirical Findings:**
- **Systematic Improvement**: All offline classification models demonstrated statistically significant improvements over online PVC baseline performance
- **Deep Learning Performance**: EEGNet achieved superior performance on raw EEG signals, validating end-to-end learning approaches for EEG classification
- **Feature Engineering Impact**: CSP-based models (CSP+LDA, CSP+MLP, CSP+RF, CSP+XGBoost) showed consistent improvements, demonstrating the value of domain-informed feature extraction
- **Ensemble Methods**: XGBoost with CSP features achieved competitive performance, highlighting the effectiveness of gradient boosting for structured feature spaces

### Visualization & Interpretability Analysis

1. **Spatial Pattern Analysis** (`CSP_outputs/csp_topomaps.png`)
   - Topographic maps visualizing CSP filter weights across scalp
   - Validates physiologically plausible sensorimotor cortex activation patterns
   - Enables interpretation of learned spatial features

2. **Comparative Performance Analysis** (`report_figures/pvc_vs_all_models.png`)
   - Comprehensive bar plot comparing all models against online baseline
   - Error bars representing cross-validation standard deviation
   - Enables statistical comparison of model performance

3. **Multi-Dimensional Performance Profiling** (`report_figures/combined_radar_plot.png`)
   - Radar plot visualization across experimental conditions
   - Identifies model-specific strengths and weaknesses
   - Facilitates model selection based on task requirements

4. **Training Dynamics** (`eegnet_training_plots/`)
   - Per-fold and ensemble-averaged training/validation curves
   - Demonstrates convergence behavior and generalization patterns
   - Enables detection of overfitting and optimization issues

5. **Data Quality Assessment** (`report_figures/trial_outcomes_distribution.png`)
   - Distribution analysis of trial outcomes (hits, misses, aborts)
   - Provides context for online performance interpretation
   - Informs data quality filtering strategies

---

## Usage & Reproducibility

### Prerequisites

```bash
Python 3.8+
```

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

#### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook final_smr_eeg_bci_pipeline.ipynb
```

#### Option 2: Python Script
```bash
python final_pipeline.py
```

**Note**: The first run will cache MATLAB data loading for faster subsequent runs.

### Key Parameters

- **CSP Pairs**: `6` (12 total features)
- **Cross-Validation Folds**: `5`
- **EEGNet Epochs**: `800`
- **MLP Epochs**: `200`
- **Bandpass Filter**: `8-30 Hz`
- **Downsampling Factor**: `5` (1000 Hz → 200 Hz)

---

## Implementation Details

### `CSP.py`
Complete CSP implementation including:
- `csp_fit()`: Learn spatial filters from class data
- `csp_transform()`: Extract CSP features from trials
- `plot_csp_feature_extraction()`: Visualize feature separation
- `fbcsp_fit()`: Filter Bank CSP (multi-band extension)

### `LDA.py`
Linear Discriminant Analysis classifier:
- `lda_fit()`: Train LDA on features
- `lda_predict()`: Classify new samples

### `final_pipeline.py` / `final_smr_eeg_bci_pipeline.ipynb`
Complete analysis pipeline:
1. Data loading and caching
2. Online performance metrics (PVC, PTC)
3. Preprocessing pipeline
4. CSP feature extraction
5. Model training and evaluation
6. Visualization generation

---

## Experimental Protocol & Dataset

### Dataset Description
- **Cohort**: 2 participants, 2 sessions per participant
- **Experimental Paradigm**: Left-Right (LR) and Up-Down (UD) motor imagery tasks
- **Electrode Configuration**: 27 gelled electrodes following 10-20 system (F3, F4, FC5, FC3, FC1, FCz, FC2, FC4, FC6, T7, C5, C3, C1, Cz, C2, C4, C6, T8, CP5, CP3, CP1, CPz, CP2, CP4, CP6, P3, P4)
- **Sampling Rate**: 1000 Hz (downsampled to 200 Hz post-preprocessing)
- **Trial Characteristics**: Variable duration with minimum 1.5s threshold for quality control

**Domain Expertise Integration**: Direct participation in data collection provided critical insight into signal quality variations, artifact characteristics, and trial-level quality assessment. This domain knowledge informed principled preprocessing design decisions, particularly in artifact handling strategies and trial filtering criteria, demonstrating the value of domain expertise in machine learning pipeline development.

### Preprocessing Rationale
- **CAR**: Removes common artifacts and reference electrode effects
- **8-30 Hz Bandpass**: Captures mu (8-13 Hz) and beta (13-30 Hz) rhythms associated with motor imagery
- **Downsampling**: Reduces computational load while preserving SMR information
- **Trial Filtering**: Ensures sufficient data for reliable SMR detection

---

## Research Contributions & Technical Innovations

1. **End-to-End Pipeline Architecture**: Developed modular, extensible framework with clear separation of concerns (preprocessing, feature extraction, classification), enabling systematic experimentation and model comparison

2. **Reproducible Research Framework**: Implemented comprehensive reproducibility controls including deterministic random seeding, data caching mechanisms, and version-controlled experimental protocols

3. **Systematic Model Benchmarking**: Conducted rigorous comparative evaluation of 5 distinct classification paradigms with consistent preprocessing and evaluation protocols, providing empirical evidence for model selection in motor imagery BCI applications

4. **Advanced Visualization & Interpretability**: Created publication-quality visualization suite enabling spatial pattern interpretation, performance comparison, and model behavior analysis

5. **Domain-Informed Preprocessing**: Leveraged first-hand data collection experience to inform preprocessing design decisions, demonstrating integration of domain expertise with machine learning methodology

6. **Production-Grade Implementation**: Developed maintainable, well-documented codebase following software engineering best practices, ensuring long-term usability and extensibility

---

## 📚 References & Background

This project implements standard BCI methodologies:
- **CSP**: Blankertz et al. (2008) - "The BCI Competition III"
- **EEGNet**: Lawhern et al. (2018) - "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
- **Motor Imagery**: Pfurtscheller & Neuper (2001) - "Motor imagery and direct brain-computer communication"

---

## 📝 Notes

- **Data**: Original MATLAB data files are required in `ORIGINAL_DATA_Group1_BCI_SMR/`
- **Caching**: First run creates `bci_session_data_cache.pkl` for faster subsequent loads
- **Outputs**: All figures are saved to respective directories with 300 DPI resolution
- **Compatibility**: Tested on Python 3.8+ with Windows/Linux

---

## Author

**Micah Baldonado**  
Carnegie Mellon University  
Fall 2025

---

## 📄 License

This project is part of academic coursework. Please respect data usage agreements and cite appropriately if using this code.

---

## 🙏 Acknowledgments

- CMU BCI Course Instructors
- Group 1 BCI SMR Dataset Contributors
- Open-source BCI community for methodology references

