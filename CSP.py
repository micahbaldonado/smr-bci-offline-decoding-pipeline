import numpy as np 
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def _concat_cov(epochs, demean=True):
    """
    estimate class covariance on concatenated epochs.
    epochs: list of arrays, each (n_chans, n_samples)
    Returns: (n_chans, n_chans) covariance.
    """
    # Concatenate along time
    X = np.concatenate([
        (e - e.mean(axis=-1, keepdims=True)) if demean else e
        for e in epochs
    ], axis=1)  # (n_ch, total_T)
    # Sample covariance 
    C = (X @ X.T) / (X.shape[1] - 1)
    # Symmetrize (numerical hygiene)
    C = 0.5 * (C + C.T)
    return C


def csp_fit(class1_epochs, class2_epochs, reg=1e-6, m_pairs=3):
    """
    Fit CSP filters:
      - covariance per class from concatenated epochs
      - generalized eigendecomposition: C1 v = λ (C1 + C2) v
      - two-class component order by descending |λ - 0.5|
    Returns:
      W: filters, shape (n_ch, 2*m_pairs)
    """
    C1 = _concat_cov(class1_epochs)
    C2 = _concat_cov(class2_epochs)
    Csum = C1 + C2

    n_ch = C1.shape[0]
    # Light ridge to ensure positive-definiteness
    scale1 = np.trace(C1) / n_ch
    scales = np.trace(Csum) / n_ch
    C1r  = C1  + reg * scale1 * np.eye(n_ch)
    Csumr = Csum + reg * scales * np.eye(n_ch)

    # Solve generalized eigenproblem C1 v = λ (C1 + C2) v
    lam, V = eigh(C1r, Csumr)
    # we want to maximize the variance ratio betweeen C1 and C1 + C2 but scaled

    # goal of CSP: find filters that maximize the variance in one class while minimizing it in the other
    # the most discriminative filters have λ far from 0.5
    order = np.argsort(np.abs(lam - 0.5))[::-1] # order by decreasing |λ - 0.5|
    V = V[:, order]

    # Take top 2*m_pairs components
    # W = V[:2*n_ch/2]   # filters
    W = np.hstack([V[:, :m_pairs], V[:, -m_pairs:]])
    # Each column is one filter, so we take the most discriminative ones from both ends
    # and stack them side by side

    
    return W


def csp_transform(W, epochs, log=True):
    """
    Project epochs with CSP filters and return average band power with optional log-transform.
    epochs: list of (n_ch, n_samp)
    Returns: (n_trials, n_components)
    """
    feats = []
    for e in epochs:
        Z = W.T @ e        # Project epochs with CSP filters (n_comp, T)
        p = (Z ** 2).mean(axis=1)  # average power
        if log: # log-transform (optional)
            p = np.log(p + 1e-12) 
        feats.append(p)
    return np.asarray(feats)


def plot_csp_feature_extraction(
    epochs, y,
    W,                 # CSP filters 
    log_power=True
):
    """
    Visualize CSP feature extraction on 2-class data.

    Parameters
    ----------
    epochs : list[np.ndarray]
        Each trial array is (n_ch, n_time), already preprocessed & windowed.
    y : array-like of shape (n_trials,)
        Binary labels (e.g., {1,2}).
    log_power : bool
        Apply log to average power (both panels) for consistency.

    Returns
    -------
    fig : matplotlib Figure
    (X_raw, X_csp) : tuple of np.ndarray
        2D raw powers and 2D CSP powers used for plotting.
    """
    classes = np.unique(y)

    # ch1, ch2 = (25, 29) # raw channels to show before CSP; here we choose C3 and C4 for visualization
    ch1, ch2 = (0, 1) # we need to adjust this since we are only looking at specific channels now.

    # -------- Raw 2D features (mean power in two chosen channels) --------
    X_raw = []
    for e in epochs:
        p1 = (e[ch1] ** 2).mean()
        p2 = (e[ch2] ** 2).mean()
        if log_power:
            p1 = np.log(p1 + 1e-12)
            p2 = np.log(p2 + 1e-12)
        X_raw.append([p1, p2])
    X_raw = np.asarray(X_raw)

    # -------- CSP 2D features (first two CSP components) --------
    feats = csp_transform(W, epochs, log=log_power)  # (n_trials, n_comp)
    if feats.shape[1] < 2:
        # pad to 2D if user only asked for 1 component
        feats = np.pad(feats, ((0, 0), (0, 2 - feats.shape[1])), constant_values=0.0)
    X_csp = feats[:, :2]

    # -------- Plotting --------
    title_before="Before CSP filtering" # i went ahead and removed the commas in case there would be and error.
    title_after="After CSP filtering"
    figsize=(6, 9)
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    # Before CSP
    ax = axes[0]
    for i, c in enumerate(classes):
        m = (y == c)
        ax.scatter(X_raw[m, 0], X_raw[m, 1],
                   s=18, alpha=0.75, marker='x' if i == 0 else 'o',
                   label=f"class {c}")
    ax.set_title(title_before)
    ax.set_xlabel(f"Channel {ch1+1} power")
    ax.set_ylabel(f"Channel {ch2+1} power")
    ax.legend(loc="best")

    # After CSP
    ax = axes[1]
    for i, c in enumerate(classes):
        m = (y == c)
        ax.scatter(X_csp[m, 0], X_csp[m, 1],
                   s=18, alpha=0.75, marker='x' if i == 0 else 'o',
                   label=f"class {c}")
    ax.set_title(title_after)
    ax.set_xlabel("CSP component 1 power")
    ax.set_ylabel("CSP component 2 power")
    ax.legend(loc="best")

    fig.tight_layout()
    return fig, (X_raw, X_csp)


def fbcsp_fit(class1_epochs, class2_epochs, frequency_bands, reg=1e-6, m_pairs=3, fs=200):
    """
    Fit Filter Bank CSP (FBCSP) filters for multiple frequency bands.
    
    Parameters
    ----------
    class1_epochs : list of np.ndarray
        Each array is (n_chans, n_samples) for class 1
    class2_epochs : list of np.ndarray
        Each array is (n_chans, n_samples) for class 2
    frequency_bands : list of tuples
        List of (low, high) frequency band tuples, e.g., [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28)]
    reg : float
        Regularization parameter for CSP (default: 1e-6)
    m_pairs : int
        Number of CSP filter pairs per band (default: 3)
    fs : float
        Sampling frequency in Hz (default: 200, assuming downsampled data)
    
    Returns
    -------
    filters_dict : dict
        Dictionary with keys as frequency band strings (e.g., "8-12") and values as CSP filters W
        Each W has shape (n_ch, 2*m_pairs)
    """
    from scipy import signal
    
    filters_dict = {}
    
    for low, high in frequency_bands:
        band_key = f"{low}-{high}"
        
        # Apply bandpass filter to each epoch in both classes
        class1_filtered = []
        class2_filtered = []
        
        # Filter class1 epochs
        for epoch in class1_epochs:
            # Apply bandpass filter
            padded = np.pad(epoch, pad_width=((0, 0), (50, 50)), mode='constant', constant_values=0)
            b, a = signal.butter(4, [low, high], btype='bandpass', fs=fs)
            filtered = signal.lfilter(b, a, padded, axis=1)
            filtered = filtered[:, 50:-50]  # Remove padding
            class1_filtered.append(filtered)
        
        # Filter class2 epochs
        for epoch in class2_epochs:
            padded = np.pad(epoch, pad_width=((0, 0), (50, 50)), mode='constant', constant_values=0)
            b, a = signal.butter(4, [low, high], btype='bandpass', fs=fs)
            filtered = signal.lfilter(b, a, padded, axis=1)
            filtered = filtered[:, 50:-50]  # Remove padding
            class2_filtered.append(filtered)
        
        # Fit CSP for this frequency band
        W = csp_fit(class1_filtered, class2_filtered, reg=reg, m_pairs=m_pairs)
        filters_dict[band_key] = W
    
    return filters_dict


def fbcsp_transform(filters_dict, epochs, log=True):
    """
    Transform epochs using Filter Bank CSP filters.
    
    Parameters
    ----------
    filters_dict : dict
        Dictionary from fbcsp_fit() with frequency band keys and CSP filter values
    epochs : list of np.ndarray
        Each array is (n_chans, n_samples)
    log : bool
        Apply log-transform to power features (default: True)
    
    Returns
    -------
    features : np.ndarray
        Shape (n_trials, n_bands * n_components_per_band)
        Features concatenated across all frequency bands
    """
    all_features = []
    
    for epoch in epochs:
        band_features = []
        
        for band_key, W in filters_dict.items():
            # Extract CSP features for this band
            # Note: We apply the filter directly since epochs are already in time domain
            # The filtering was done during fit, but here we need to filter again
            # Actually, wait - in FBCSP, we filter during fit to learn filters,
            # but during transform, we should filter the test data too!
            # However, the filters W were learned on filtered data, so we need to filter here too.
            # But actually, a common approach is to filter during fit AND transform.
            # For simplicity, let's assume epochs are already filtered appropriately,
            # or we filter them here. Actually, let me check the standard FBCSP approach...
            
            # Standard FBCSP: filter during fit to learn filters, filter during transform to apply
            # But we don't have fs here easily... Let's make it simpler:
            # Apply CSP transform directly (assuming data is already in the right frequency range)
            # OR we need to pass fs and frequency_bands to this function too.
            
            # For now, let's apply CSP transform directly (user should filter epochs before calling)
            # This is actually the cleaner approach - filter in the pipeline, then call transform
            Z = W.T @ epoch  # Project with CSP filters
            p = (Z ** 2).mean(axis=1)  # Average power
            if log:
                p = np.log(p + 1e-12)
            band_features.append(p)
        
        # Concatenate features from all bands
        all_features.append(np.concatenate(band_features))
    
    return np.asarray(all_features)


def fbcsp_fit_transform(class1_epochs, class2_epochs, frequency_bands, reg=1e-6, m_pairs=3, fs=200, log=True):
    """
    Convenience function: Fit FBCSP filters and transform training data.
    This filters the data during both fit and transform.
    
    Parameters
    ----------
    class1_epochs : list of np.ndarray
        Each array is (n_chans, n_samples) for class 1
    class2_epochs : list of np.ndarray
        Each array is (n_chans, n_samples) for class 2
    frequency_bands : list of tuples
        List of (low, high) frequency band tuples
    reg : float
        Regularization parameter for CSP
    m_pairs : int
        Number of CSP filter pairs per band
    fs : float
        Sampling frequency in Hz
    log : bool
        Apply log-transform to power features
    
    Returns
    -------
    filters_dict : dict
        Dictionary with frequency band keys and CSP filter values
    features_class1 : np.ndarray
        FBCSP features for class1 epochs
    features_class2 : np.ndarray
        FBCSP features for class2 epochs
    """
    filters_dict = fbcsp_fit(class1_epochs, class2_epochs, frequency_bands, reg, m_pairs, fs)
    
    # Transform both classes
    features_class1 = fbcsp_transform_with_filtering(filters_dict, class1_epochs, frequency_bands, fs, log)
    features_class2 = fbcsp_transform_with_filtering(filters_dict, class2_epochs, frequency_bands, fs, log)
    
    return filters_dict, features_class1, features_class2


def fbcsp_transform_with_filtering(filters_dict, epochs, frequency_bands, fs=200, log=True):
    """
    Transform epochs using FBCSP filters, applying bandpass filtering for each band.
    This is the proper FBCSP transform that filters data for each band.
    
    Parameters
    ----------
    filters_dict : dict
        Dictionary from fbcsp_fit() with frequency band keys and CSP filter values
    epochs : list of np.ndarray
        Each array is (n_chans, n_samples) - raw (unfiltered) epochs
    frequency_bands : list of tuples
        List of (low, high) frequency band tuples (must match filters_dict keys)
    fs : float
        Sampling frequency in Hz
    log : bool
        Apply log-transform to power features
    
    Returns
    -------
    features : np.ndarray
        Shape (n_trials, n_bands * n_components_per_band)
    """
    from scipy import signal
    
    all_features = []
    
    for epoch in epochs:
        band_features = []
        
        # Ensure order matches by iterating frequency_bands and looking up corresponding filter
        for low, high in frequency_bands:
            band_key = f"{low}-{high}"
            W = filters_dict[band_key]  # Look up filter by key to ensure correct order
            
            # Apply bandpass filter for this frequency band
            padded = np.pad(epoch, pad_width=((0, 0), (50, 50)), mode='constant', constant_values=0)
            b, a = signal.butter(4, [low, high], btype='bandpass', fs=fs)
            filtered = signal.lfilter(b, a, padded, axis=1)
            filtered = filtered[:, 50:-50]  # Remove padding
            
            # Apply CSP transform
            Z = W.T @ filtered  # Project with CSP filters
            p = (Z ** 2).mean(axis=1)  # Average power
            if log:
                p = np.log(p + 1e-12)
            band_features.append(p)
        
        # Concatenate features from all bands
        all_features.append(np.concatenate(band_features))
    
    return np.asarray(all_features)
