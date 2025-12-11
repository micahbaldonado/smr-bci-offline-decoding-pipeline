# Filter Bank CSP (FBCSP) Implementation Guide

## Difficulty Assessment: **MODERATE** ⚙️

**Estimated Time:** 3-4 hours  
**Complexity:** Medium - Requires understanding filter banks and feature concatenation

---

## What is FBCSP?

Filter Bank CSP extends standard CSP by:
1. **Applying multiple bandpass filters** to the EEG data (e.g., 8-12 Hz, 12-16 Hz, 16-20 Hz, etc.)
2. **Running CSP independently** on each filtered frequency band
3. **Concatenating features** from all bands into a single feature vector

**Why it works better:** Different frequency bands capture different aspects of motor imagery. FBCSP captures discriminative information across the entire frequency spectrum.

---

## What Has Been Added

✅ **New functions in `CSP.py`:**
- `fbcsp_fit()` - Fits CSP filters for multiple frequency bands
- `fbcsp_transform_with_filtering()` - Transforms data using FBCSP (applies filtering + CSP)
- `fbcsp_fit_transform()` - Convenience function for training data

---

## Required Changes to Your Pipeline

### Step 1: Import FBCSP Functions

**Add to your imports:**
```python
from CSP import csp_fit, csp_transform, plot_csp_feature_extraction
from CSP import fbcsp_fit, fbcsp_transform_with_filtering  # ADD THIS LINE
```

### Step 2: Define Frequency Bands

**Add this BEFORE your loop:**
```python
# Define frequency bands for FBCSP (typical SMR bands)
frequency_bands = [
    (8, 12),   # Alpha band
    (12, 16),  # Low beta
    (16, 20),  # Mid beta  
    (20, 24),  # High beta
    (24, 28),  # Upper beta
    (28, 32)   # Gamma (optional)
]

# Or use overlapping bands (common in FBCSP):
# frequency_bands = [
#     (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)
# ]

fs = 200  # Your downsampled sampling rate (1000 Hz / 5 = 200 Hz)
```

### Step 3: Modify Your Loop

**REPLACE the CSP section with FBCSP:**

```python
for exp_name, trial_list in all_data.items():

    # extract EEG trials and labels for THIS experiment
    eeg_trial_data = [trial["eeg"] for trial in trial_list]
    labels = [trial["target"] for trial in trial_list]

    X_train, X_test, y_train, y_test, = train_test_split(
        eeg_trial_data, labels, 
        test_size=1/5, random_state=13, shuffle=True)

    # prepare class splits (on training data only)
    X1 = [tr for tr, lab in zip(X_train, y_train) if lab == 1]
    X2 = [tr for tr, lab in zip(X_train, y_train) if lab == 2]

    # ===== FBCSP SECTION =====
    csp_pairs = 6  # Number of CSP pairs per frequency band
    
    # Fit FBCSP filters (learns filters for each frequency band)
    fbcsp_filters = fbcsp_fit(
        X1, X2, 
        frequency_bands=frequency_bands,
        reg=1e-6, 
        m_pairs=csp_pairs,
        fs=fs
    )
    
    # Transform training and test data
    # Note: This function filters the data for each band internally
    fbcsp_train = fbcsp_transform_with_filtering(
        fbcsp_filters, X_train, frequency_bands, fs=fs, log=True
    )
    fbcsp_test = fbcsp_transform_with_filtering(
        fbcsp_filters, X_test, frequency_bands, fs=fs, log=True
    )
    
    # ===== OPTIONAL: Keep original CSP for comparison =====
    W = csp_fit(X1, X2, reg=1e-6, m_pairs=csp_pairs)
    f_train = csp_transform(W, X_train)
    f_test = csp_transform(W, X_test)
    
    # Visualize (using original CSP for visualization)
    fig_tmp, _ = plot_csp_feature_extraction(
        epochs=X_train,
        y=y_train,
        W=W,
        log_power=True
    )
    fig_tmp.suptitle(f"CSP Visualization: {exp_name}")
    fig_tmp.savefig(f"CSP_outputs/{exp_name}.png", dpi=300)
    plt.close(fig_tmp)

    # store the results
    results[exp_name] = {
        "W": W,                    # Original CSP filters (optional)
        "fbcsp_filters": fbcsp_filters,  # FBCSP filters dict
        "X_train": X_train,
        "X_test": X_test,
        "f_train": f_train,        # Original CSP features (optional)
        "f_test": f_test,          # Original CSP features (optional)
        "fbcsp_train": fbcsp_train,  # ✨ NEW: FBCSP training features
        "fbcsp_test": fbcsp_test,    # ✨ NEW: FBCSP test features
        "y_train": y_train,
        "y_test": y_test,
    }
```

---

## Feature Dimensions

**Original CSP:**
- Features: `(n_trials, 2 * csp_pairs)` = `(n_trials, 12)` if `csp_pairs=6`

**FBCSP:**
- Features: `(n_trials, n_bands * 2 * csp_pairs)`
- Example: 6 bands × 12 features = `(n_trials, 72)` features

**Note:** FBCSP gives you more features, which can improve classification but may require regularization.

---

## Key Differences from Standard CSP

| Aspect | Standard CSP | FBCSP |
|--------|-------------|-------|
| **Frequency filtering** | Single band (8-30 Hz) | Multiple bands (e.g., 6 bands) |
| **CSP application** | Once on filtered data | Once per frequency band |
| **Feature count** | `2 * m_pairs` | `n_bands * 2 * m_pairs` |
| **Computational cost** | Low | Moderate (6x more CSP operations) |
| **Typical performance** | Good | Usually better |

---

## Tips for Optimization

1. **Start with fewer bands** (e.g., 4 bands: 8-12, 12-16, 16-20, 20-24) to test
2. **Adjust `csp_pairs`** - You might need fewer pairs per band (e.g., 3-4) since you have more total features
3. **Feature selection** - Consider using mutual information-based feature selection to reduce dimensionality
4. **Sampling rate** - Make sure `fs=200` matches your downsampled rate (1000 Hz / 5 = 200 Hz)

---

## Testing Your Implementation

After implementing, verify:
1. ✅ `fbcsp_train.shape[1] == len(frequency_bands) * 2 * csp_pairs`
2. ✅ `fbcsp_test.shape[0] == len(X_test)`
3. ✅ Features are not all zeros or NaNs
4. ✅ Classification accuracy improves (or at least doesn't degrade)

---

## Example Usage After Implementation

```python
# Access FBCSP features
fbcsp_train = results['S1_LR']['fbcsp_train']
fbcsp_test = results['S1_LR']['fbcsp_test']

# Use with your classifier (LDA, etc.)
from LDA import lda_fit, lda_predict
lda_model = lda_fit(fbcsp_train, y_train)
predictions = lda_predict(lda_model, fbcsp_test)
```

---

## Summary

**What you need to do:**
1. ✅ FBCSP functions already added to `CSP.py`
2. ⚠️ Add import: `from CSP import fbcsp_fit, fbcsp_transform_with_filtering`
3. ⚠️ Define `frequency_bands` list
4. ⚠️ Replace CSP calls with FBCSP calls in your loop
5. ⚠️ Add `fbcsp_train` and `fbcsp_test` to results dictionary

**Difficulty:** Moderate - The functions are ready, you just need to integrate them into your pipeline!

