# preprocessing.py
import numpy as np
from scipy import signal
from scipy.signal import resample

# The 27 gelled channels from the experiment (in order they appear in the data)
GELED_CHANNELS = ['F3','F4','FC5','FC3','FC1','FCz','FC2','FC4','FC6',
                  'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
                  'CP5','CP3','CP1','CPz','CP2','CP4','CP6','P3','P4']


def get_geled_channel_indices(runData):
    """
    Find the indices of the 27 gelled channels in the runData.label array.
    Returns the indices in the original data array that correspond to gelled channels.
    """
    if not hasattr(runData, 'label'):
        raise ValueError("runData must have 'label' attribute to identify gelled channels")
    
    labels = runData.label
    gelled_indices = []
    
    # Convert labels to list of strings for comparison
    label_list = [str(labels[i]).strip() for i in range(len(labels))]
    
    # Find indices of gelled channels
    for gelled_name in GELED_CHANNELS:
        # Try to find matching channel (case-insensitive, handle variations)
        found = False
        for idx, label in enumerate(label_list):
            if label.upper() == gelled_name.upper():
                gelled_indices.append(idx)
                found = True
                break
        if not found:
            print(f"Warning: Gelled channel {gelled_name} not found in data labels")
    
    return gelled_indices


def find_motor_channels_from_geled(runData, geled_indices):
    """
    Find motor cortex channels from the gelled channel subset.
    Motor ROI: C5, C3, C1, C2, C4, C6 (skipping Cz).
    Returns indices within the geled_indices array (not original 68-channel indices).
    """
    if not hasattr(runData, 'label'):
        raise ValueError("runData must have 'label' attribute")
    
    labels = runData.label
    motor_channel_names = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6']
    
    motor_indices_in_geled = []
    
    for motor_name in motor_channel_names:
        # Find this motor channel in the geled channels
        for geled_idx, orig_idx in enumerate(geled_indices):
            if orig_idx < len(labels):
                label_str = str(labels[orig_idx]).strip().upper()
                if label_str == motor_name.upper():
                    motor_indices_in_geled.append(geled_idx)
                    break
    
    if len(motor_indices_in_geled) != 6:
        print(f"Warning: Expected 6 motor channels, found {len(motor_indices_in_geled)}")
        print(f"Found motor channels: {motor_indices_in_geled}")
    
    return motor_indices_in_geled


def extract_trials_from_lockin(bci_session_data, session, session_type,
                               mi_window_length=2.0):  # Fixed 2-second MI window
    """
    Extract trials with CORRECT motor imagery window: [trialStart - 2.0s, trialStart].
    
    CRITICAL: trialStart is NOT when the ball appears. According to the .mat description:
    - trialStart = time when feedback starts (cursor appears)
    - The ball (target) appears 2 seconds BEFORE trialStart
    - The actual MI period is the 2 seconds BEFORE the cursor appears
    
    This extracts exactly the motor imagery period used by BCI2000 and all MI studies.
    
    Parameters:
    -----------
    mi_window_length : float
        Length of MI window in seconds (default 2.0s, matching BCI2000)
    """
    trials = []
    labels = []
    online_preds = []
    
    run_list = bci_session_data[session][session_type]
    
    for runData in run_list:
        fs = runData.fs
        allData = runData.allData  # (n_chans, n_timepoints)
        trialStart = np.asarray(runData.trialStart).ravel()
        trialEnd = np.asarray(runData.trialEnd).ravel()
        targets = np.asarray(runData.target).ravel()
        outcomes = np.asarray(runData.outcome).ravel()
        
        # Fixed window length in samples
        window_length_samples = int(mi_window_length * fs)
        
        for i, start in enumerate(trialStart):
            # CORRECT WINDOW: [trialStart - 2.0s, trialStart]
            # This is the actual motor imagery period before cursor feedback
            window_start = start - window_length_samples
            window_end = start  # Exactly at trialStart
            
            # Check if window fits within allData bounds
            if window_start < 0 or window_end > allData.shape[1]:
                continue  # Skip if window extends beyond recording
            
            # Extract the motor imagery window (fixed 2-second length)
            data = allData[:, window_start:window_end]
            
            # Verify we got exactly the right length
            if data.shape[1] != window_length_samples:
                continue  # Skip if length doesn't match (shouldn't happen, but safety check)
            
            # Label from target (MI cue: 1=left/right, 2=right/left)
            label = int(targets[i])
            labels.append(label)
            trials.append(data)
            
            # Online prediction: label if hit, else 0
            online_pred = label if outcomes[i] == 1 else 0
            online_preds.append(online_pred)
    
    # Get sampling rate from first run
    srate = int(bci_session_data[session][session_type][0].fs)
    
    # Return as lists (all trials now have fixed length = window_length_samples)
    if len(trials) > 0:
        return trials, np.asarray(labels), np.asarray(online_preds), srate
    else:
        return [], np.array([]), np.array([]), srate


def extract_trials_from_all_sessions(bci_session_data, session_list, session_type,
                                     mi_window_length=2.0,  # Fixed 2-second MI window
                                     runData_sample=None):  # Sample runData for channel filtering
    """
    Extract trials from multiple sessions using CORRECT MI window: [trialStart - 2.0s, trialStart].
    
    All trials are fixed-length (2 seconds = 2000 samples at 1000 Hz).
    Filters to only the 27 gelled channels before returning.
    
    Parameters:
    -----------
    mi_window_length : float
        Length of MI window in seconds (default 2.0s)
    runData_sample : runData object
        Sample runData to identify gelled channels (required for channel filtering)
    """
    all_trials = []
    all_labels = []
    all_online_preds = []
    
    # Get geled channel indices from sample runData
    if runData_sample is None:
        # Try to get from first available session
        for session in session_list:
            if len(bci_session_data[session][session_type]) > 0:
                runData_sample = bci_session_data[session][session_type][0]
                break
    
    if runData_sample is None:
        raise ValueError("Cannot determine gelled channels: no runData available")
    
    geled_indices = get_geled_channel_indices(runData_sample)
    print(f"Found {len(geled_indices)} gelled channels (out of {runData_sample.allData.shape[0]} total channels)")
    
    # First pass: extract all trials (all should be fixed-length now)
    for session in session_list:
        trials, labels, online_preds, srate = extract_trials_from_lockin(
            bci_session_data,
            session=session,
            session_type=session_type,
            mi_window_length=mi_window_length
        )
        
        if len(trials) > 0:
            # Filter to only gelled channels
            filtered_trials = []
            for trial in trials:
                # trial shape: (n_chans, n_samples) - filter to geled channels
                filtered_trial = trial[geled_indices, :]
                filtered_trials.append(filtered_trial)
            
            all_trials.extend(filtered_trials)
            all_labels.extend(labels)
            all_online_preds.extend(online_preds)
    
    if len(all_trials) == 0:
        return np.array([]), np.array([]), np.array([]), 1000
    
    # Verify all trials have the same length (they should, since we use fixed window)
    trial_lengths = [t.shape[1] for t in all_trials]
    expected_length = int(mi_window_length * srate)
    
    if not all(tl == expected_length for tl in trial_lengths):
        print(f"Warning: Not all trials have expected length {expected_length}")
        print(f"  Trial lengths: min={min(trial_lengths)}, max={max(trial_lengths)}, expected={expected_length}")
        # Truncate all to minimum length (shouldn't be necessary, but safety check)
        min_length = min(trial_lengths)
        all_trials = [t[:, :min_length] for t in all_trials]
        print(f"  Truncated all trials to {min_length} samples")
    
    # Convert to array (all trials now have same shape: n_trials, n_geled_chans, n_samples)
    trials_array = np.asarray(all_trials)
    labels_array = np.asarray(all_labels)
    online_preds_array = np.asarray(all_online_preds)
    
    print(f"Extracted {len(all_trials)} trials")
    print(f"  Trial shape: {trials_array.shape}")
    print(f"  Window length: {trials_array.shape[2]/srate:.2f} seconds (fixed {mi_window_length}s MI period)")
    print(f"  Channels: {trials_array.shape[1]} (gelled channels only)")
    
    return trials_array, labels_array, online_preds_array, srate


def preprocess_trials(trials, srate, channel_indices=None, runData_sample=None):
    """
    Preprocessing pipeline matching CSP_LDA_Assignment.py lines 52-77.
    
    Order: CAR -> Downsample -> Filter -> Channel Selection
    
    IMPORTANT: trials input should already be filtered to gelled channels only.
    channel_indices refer to indices WITHIN the gelled channel array (not original 68 channels).
    
    Parameters
    ----------
    trials : np.ndarray
        Shape (n_trials, n_geled_chans, n_samples) - should be 27 gelled channels, 2000 samples
    srate : int
        Sampling rate (should be 1000 Hz)
    channel_indices : list or None
        Channel indices WITHIN the gelled channel array. If None, auto-detect motor channels
    runData_sample : runData object or None
        Sample runData to detect motor channels from gelled subset
    """
    # Get motor channel indices (within gelled channel array)
    if channel_indices is None:
        if runData_sample is not None:
            geled_indices = get_geled_channel_indices(runData_sample)
            channel_indices = find_motor_channels_from_geled(runData_sample, geled_indices)
        else:
            # Fallback: assume motor channels are at specific positions in gelled array
            # GELED_CHANNELS = ['F3','F4','FC5','FC3','FC1','FCz','FC2','FC4','FC6',
            #                   'T7','C5','C3','C1','Cz','C2','C4','C6','T8',...]
            # Motor channels: C5=10, C3=11, C1=12, C2=14, C4=15, C6=16 (0-indexed)
            channel_indices = [10, 11, 12, 14, 15, 16]  # Indices in gelled array
    
    # Step 1: Common average referencing (matching CSP_LDA_Assignment.py line 55)
    # CAR across channels for each trial
    trials = trials - trials.mean(axis=1, keepdims=True)
    
    # Step 2: Downsample (matching CSP_LDA_Assignment.py lines 58-62)
    downsrate = 100
    n_trials, n_chans, n_samples = np.shape(trials)
    newSamples = int(n_samples/srate*downsrate)
    trials = resample(trials, newSamples, t=None, axis=-1, window=None, domain='time')
    
    # Step 3: Bandpass filtering [4, 40] Hz (matching CSP_LDA_Assignment.py lines 64-70)
    padding_length = 100
    padded_sig = np.pad(trials, ((0,0),(0,0),(padding_length,padding_length)), 
                       'constant', constant_values=0)
    b, a = signal.butter(4, [4, 40], btype='bandpass', fs=downsrate)
    padded_sig = signal.lfilter(b, a, padded_sig, axis=-1)
    trials = padded_sig[:,:,padding_length:-padding_length]
    
    # Step 4: Channel selection - select motor ROI from gelled channels
    # channel_indices now refer to positions within the gelled channel array
    if len(channel_indices) <= trials.shape[1]:
        valid_indices = [idx for idx in channel_indices if idx < trials.shape[1]]
        if len(valid_indices) > 0:
            trials = trials[:, valid_indices, :]
        else:
            raise ValueError(f"Invalid channel indices {channel_indices} for {trials.shape[1]} gelled channels")
    else:
        raise ValueError(f"Too many channel indices {len(channel_indices)} for {trials.shape[1]} gelled channels")
    
    return trials
