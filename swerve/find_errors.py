import numpy as np

# This function will input the raw GIC data and determine if there is an error with the timeseries. If there is, it will log the error and output it
# will be added to info.py and run before metrics are calculated

def find_errors(data, sid, data_source, logger=None, spike_filt_type='both'):
    """low_signal_threshold, baseline_buffer, spike_threshold, and std_limit in [A]
       max_cadence, max_gap, and max_constant in [s]
       Returns a list of detected error messages (empty list if none)."""
    from swerve import config, cadence
    from datetime import timedelta
    CONFIG = config()
    gic_filter_kwargs = CONFIG['find_errors_kwargs']

    errors = []

    # Removing NERC sites that are TVA duplicates
    if data_source == 'NERC' and 'sid_duplicates' in CONFIG and sid in CONFIG['sid_duplicates']:
        errors.append(f"Duplicate of TVA site '{CONFIG['sid_duplicates'][sid]}'")

    data_meas = data['data']
    time_meas = data['time']

    # Removing any sites with all negative or all positive values
    if np.all(data_meas >= 0):
        errors.append("All GIC values are positive")
    if np.all(data_meas <= 0):
        errors.append("All GIC values are negative")

    # Removing sites with low signal (all values within +/- low_signal_threshold [A])
    low_signal_threshold = gic_filter_kwargs['low_signal_threshold']
    if np.all((-low_signal_threshold <= data_meas) & (data_meas <= low_signal_threshold)):
        errors.append(f"Low signal: all GIC values within +/- {low_signal_threshold} A")

    # Removing all sites with baseline offset [A]
    median_val = np.median(data_meas)
    baseline_buffer = gic_filter_kwargs['baseline_buffer']
    if median_val > baseline_buffer or median_val < -baseline_buffer:
        errors.append(f"Baseline offset detected (median value: {median_val} A)")

    # Removing sites with large, unphysical spikes (> spike_threshold [A])
    spike_threshold = gic_filter_kwargs['spike_threshold']
    data_array = np.array(data_meas).ravel()
    diffs = np.abs(np.diff(data_array))
    if spike_filt_type == 'difference' or spike_filt_type == 'both':
        _diff_spike_filt(diffs, spike_threshold, errors)
    elif spike_filt_type == 'median' or spike_filt_type == 'both':
        median_window = gic_filter_kwargs['median_window']
        _median_spike_filt(data_array, spike_threshold, errors, win=median_window)
    else:
        raise ValueError(f"Unknown spike_filt_type: {spike_filt_type}")

    # Removing noisy sites before storm (std before > 1/noise_threshold * std after)
    storm_start = CONFIG['limits']['data'][0]
    storm_stop = CONFIG['limits']['data'][1]
    noise_threshold = gic_filter_kwargs['noise_threshold']
    try:
        pre_mask = np.array([(t >= time_meas[0]) and (t < storm_start) for t in time_meas])
        post_mask = np.array([(t >= storm_start) and (t < storm_stop) for t in time_meas])
    except Exception:
        # Fallback: use POSIX seconds if storm_start is a numeric timestamp
        time_secs = np.array([t.timestamp() for t in time_meas])
        ss = float(storm_start)
        pre_mask = (time_secs >= ss - 2*3600) & (time_secs < ss)
        post_mask = (time_secs >= ss) & (time_secs < ss + 2*3600)
    std_pre = float(np.std(data_array[pre_mask]))
    std_post = float(np.std(data_array[post_mask]))
    # if std_pre > 1/noise_threshold * std_post then flag as noisy before storm
    if std_post == 0.0:
        if std_pre > 0.0:
            errors.append("Excessive pre-storm noise: nonzero std before while std after is zero")
    else:
        if noise_threshold*std_pre > std_post:
            errors.append(f"Excessive pre-storm noise: std before ({std_pre:.4f} A) > 1/{noise_threshold} * std after ({std_post:.4f} A)")

    # Removing any sites with dt >= max_cadence [s] or with gap in data >= max_gap [s]
    from swerve import subset
    crop_time_meas, crop_data_meas = subset(time_meas, data_meas, storm_start, storm_stop)
    dt = cadence(crop_time_meas, logger=logger, logger_indent=2) # returns cadence in ns
    dt_array = (np.array(dt)).astype(np.float64)
    if dt_array.size:
        max_cadence = gic_filter_kwargs['max_cadence']
        max_gap = gic_filter_kwargs['max_gap']
        if (any(dt_array >= max_cadence*1e9) and len(dt_array) == 1):
            errors.append(f"Cadence is >= {max_cadence} seconds ({max(dt_array)/1e9} seconds)")
        if any(dt_array >= max_gap*1e9):
            errors.append(f"Data gap detected >= {max_gap/60} minutes ({max(dt_array)/1e9} seconds)")

    # Remove sites with constant values for more than max_const [s]
    time_array = np.array([t.timestamp() for t in crop_time_meas])  # convert to seconds
    crop_data_array = np.array(crop_data_meas).ravel()
    crop_diffs = np.diff(crop_data_array)
    window_s = gic_filter_kwargs['max_const'] 
    # Find runs of constant values
    change_idx = np.where(crop_diffs != 0)[0] + 1
    if change_idx.size:
        run_starts = np.concatenate(([0], change_idx))
        run_ends = np.concatenate((change_idx, [len(crop_data_array)]))
    else:
        run_starts = np.array([0])
        run_ends = np.array([len(crop_data_array)])
    for start, end in zip(run_starts, run_ends):
        if end - start > 1:
            elapsed_s = time_array[end - 1] - time_array[start]
            if elapsed_s >= window_s:
                errors.append(f"Data is constant for at least 5 minutes starting at time {crop_time_meas[start]}")
                break

    return errors

def _diff_spike_filt(diffs, spike_threshold, errors):
    diffs = np.abs(diffs)
    if diffs.size and any(diffs > spike_threshold):
        errors.append(f"Unphysical spike detected (max diff: {np.max(diffs)} A)")

def _median_spike_filt(data_array, spike_threshold, errors, win=20):
    n = data_array.size
    if n == 0:
        return
    # moving window median filter: use window size (win)
    if win % 2 == 0:
        win -= 1
    half = win // 2
    deviations = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        median = np.median(data_array[start:end])
        deviations[i] = abs(data_array[i] - median)

    if any(deviations > spike_threshold):
        errors.append(f"Unphysical spike detected (max deviation from moving median: {np.max(deviations)} A)")