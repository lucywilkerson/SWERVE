import numpy as np
import pandas as pd

# This function will input the raw GIC data and clean for spikes and baseline offset. It will then
# determine if there is an error with the timeseries. If there is, it will log the error and output it
# will be added to info.py and run before metrics are calculated

def filter(data, logger=None):
    """low_signal_threshold, baseline_buffer, spike_threshold, and std_limit in [A]
       max_cadence, max_gap, and max_constant in [s]
       Returns a list of detected error messages (empty list if none)."""
    from swerve import config, cadence
    from datetime import timedelta
    CONFIG = config()
    gic_filter_kwargs = CONFIG['filter_kwargs']

    errors = []

    data_meas = data['data']
    time_meas = data['time']
    data_df = pd.DataFrame({'data': np.array(data_meas).ravel()}, index=pd.to_datetime(time_meas))
    
    # Correcting large, unphysical spikes (> spike_threshold [A])
    spike_threshold = gic_filter_kwargs['spike_threshold']
    spike_filt_type = gic_filter_kwargs['spike_filt_type']
    if spike_filt_type == None:
        pass
    elif spike_filt_type == 'difference':
        data_df['data'] = _diff_spike_filt(data_df['data'], spike_threshold, errors)
    elif spike_filt_type == 'median':
        median_window = gic_filter_kwargs['median_window']
        data_df['data'] = _median_spike_filt(data_df['data'], spike_threshold, errors, win=median_window)
    else:
        raise ValueError(f"Unknown spike_filt_type: {spike_filt_type}")

    # Correcting baseline offset [A]
    median_val = np.nanmedian(data_df['data'].to_numpy())
    baseline_buffer = gic_filter_kwargs['baseline_buffer']
    if median_val > baseline_buffer or median_val < -baseline_buffer:
        data_df['data'] = data_df['data'] - median_val

    data_series = data_df['data']

    # Removing any sites with all negative or all positive values
    # TODO: figure out how to deal w sites w abs GIC that's been detrended so not all pos anymore
    if np.all(data_series >= 0):
        errors.append("All GIC values are positive")
    if np.all(data_series <= 0):
        errors.append("All GIC values are negative")

    # Removing sites with low signal (all values within +/- low_signal_threshold [A])
    low_signal_threshold = gic_filter_kwargs['low_signal_threshold']
    if np.all((-low_signal_threshold <= data_series) & (data_series <= low_signal_threshold)):
        errors.append(f"Low signal: all GIC values within +/- {low_signal_threshold} A")

    # Removing noisy sites before storm (std before > 1/noise_threshold * std after)
    storm_start = CONFIG['limits']['data'][0]
    storm_stop = CONFIG['limits']['data'][1]
    noise_threshold = gic_filter_kwargs['noise_threshold']
    try:
        pre_mask = (data_df.index >= data_df.index[0]) & (data_df.index < storm_start)
        post_mask = (data_df.index >= storm_start) & (data_df.index < storm_stop)
    except Exception:
        # Fallback: use POSIX seconds if storm_start is a numeric timestamp
        time_secs = np.array([t.timestamp() for t in data_df.index])
        ss = float(storm_start)
        pre_mask = (time_secs >= ss - 2*3600) & (time_secs < ss)
        post_mask = (time_secs >= ss) & (time_secs < ss + 2*3600)
    std_pre = float(np.nanstd(data_df['data'][pre_mask].to_numpy()))
    std_post = float(np.nanstd(data_df['data'][post_mask].to_numpy()))
    # if std_pre > 1/noise_threshold * std_post then flag as noisy before storm
    if std_post == 0.0:
        if std_pre > 0.0:
            errors.append("Excessive pre-storm noise: nonzero std before while std after is zero")
    elif noise_threshold*std_pre > std_post:
            errors.append(f"Excessive pre-storm noise: std before ({std_pre:.4f} A) > 1/{noise_threshold} * std after ({std_post:.4f} A)")

    # Removing any sites with dt >= max_cadence [s] or with gap in data >= max_gap [s]
    from swerve import subset
    crop_time_meas, crop_data_meas = subset(data_df.index, data_df['data'], storm_start, storm_stop)
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
    try:
        time_array = np.asarray(crop_time_meas, dtype='datetime64[s]').astype(np.int64)
    except (TypeError, ValueError):
        time_array = np.array([t.timestamp() for t in crop_time_meas])
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
                error_time = pd.to_datetime(crop_time_meas[start]).round('s')
                errors.append(f"Data is constant for at least {int(window_s/60)} minutes starting at time {error_time}")
                break
    if errors==[]:
        errors = None

    return data_df, errors

def _expand_mask(mask, pre=2, post=8):
    # expands mask to include values before/after detected spike
    mask = mask.squeeze() if hasattr(mask, 'squeeze') else mask
    broad_mask = np.zeros(len(mask), dtype=bool)
    for i, value in enumerate(mask):
        if value:
            if i < pre:
                broad_mask[:i+post] = True
            elif i > len(mask) - post - 1:
                broad_mask[i-pre:] = True
            else:
                broad_mask[i-pre:i+post] = True
    return broad_mask

def _diff_spike_filt(data, spike_threshold, errors, diff_window=5, std_window=20, dd_thresh=10, std_thresh=1):
    series = data.squeeze()
    ddat = series.diff() #d/dt proxy
    dddat = ddat.diff().shift(-1) #d^2/dt^2 proxy, shifted to align with the middle point of the 3-point stencil
    d = abs(ddat).rolling(diff_window, center=True).sum() #magnitude of d/dt over a window
    dd = abs(dddat).rolling(diff_window, center=True).sum() #magnitude of d^2/dt^2 over a window
    d_std = abs(series).rolling(std_window, center=True).std() #local variability, helps distinguish spikes from real variation
    spike_mask = (d > spike_threshold/2) & (dd >= spike_threshold) & (d_std <= spike_threshold/2)
    if spike_mask.any():
        mask = _expand_mask(spike_mask)
        data[mask] = np.nan
    return data

def _median_spike_filt(data, spike_threshold, errors, win=20):
    data = data.squeeze() if hasattr(data, 'squeeze') else data
    n = data.size
    if n == 0:
        return data
    # moving window median filter: use window size (win)
    use_iloc = hasattr(data, 'iloc')
    if win % 2 == 0:
        win -= 1
    half = win // 2
    deviations = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = data.iloc[start:end] if use_iloc else data[start:end]
        median = np.median(window)
        value = data.iloc[i] if use_iloc else data[i]
        deviations[i] = abs(value - median)

    if any(deviations > spike_threshold):
        spike_mask = deviations > spike_threshold
        mask = _expand_mask(spike_mask)
        if use_iloc:
            data.iloc[mask] = np.nan
        else:
            data[mask] = np.nan
    return data