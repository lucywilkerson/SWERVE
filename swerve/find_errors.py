import numpy

# This function will input the raw GIC data and determine if there is an error with the timeseries. If there is, it will log the error and output it
# will be added to info.py and run before metrics are calculated

def find_errors(data, sid, data_source, logger=None, low_signal_threshold=3.5, baseline_buffer=1, spike_threshold = 40, std_limit=15, max_cadence=60, max_gap=600):
    """low_signal_threshold, baseline_buffer, spike_threshold, and std_limit in [A]
       max_cadence, max_gap, and max_constant in [s]
       Returns a list of detected error messages (empty list if none)."""
    from swerve import config, cadence
    CONFIG = config()

    errors = []

    # Removing NERC sites that are TVA duplicates
    if data_source == 'NERC' and 'sid_duplicates' in CONFIG and sid in CONFIG['sid_duplicates']:
        errors.append(f"Duplicate of TVA site '{CONFIG['sid_duplicates'][sid]}'")

    data_meas = data['data']
    time_meas = data['time']

    # Removing any sites with all negative or all positive values
    if all(i >= 0 for i in data_meas):
        errors.append("All GIC values are positive")
    if all(i <= 0 for i in data_meas):
        errors.append("All GIC values are negative")

    # Removing sites with low signal (all values within +/- low_signal_threshold [A])
    if all(-low_signal_threshold <= i <= low_signal_threshold for i in data_meas):
        errors.append(f"Low signal: all GIC values within +/- {low_signal_threshold} A")

    # Removing all sites with baseline offset [A]
    median_val = numpy.median(data_meas)
    if median_val > baseline_buffer or median_val < -baseline_buffer:
        errors.append(f"Baseline offset detected (median value: {median_val} A)")

    # Removing sites with large, unphysical spikes (> spike_threshold [A])
    data_array = numpy.array(data_meas).ravel()
    diffs = numpy.abs(numpy.diff(data_array))
    if diffs.size and any(diffs > spike_threshold):
        errors.append(f"Unphysical spike detected (max diff: {numpy.max(diffs)} A)")

    # Removing noisy sites
    if numpy.std(data_array) > std_limit:
        errors.append(f"Excessive noise detected (std dev: {numpy.std(data_meas)} A)")

    # Removing any sites with dt >= max_cadence [s] or with gap in data >= max_gap [s]
    dt = cadence(time_meas, logger=logger, logger_indent=2) # returns cadence in ns
    dt_array = (numpy.array(dt)).astype(numpy.float64)
    if dt_array.size:
        if (any(dt_array >= max_cadence*1e9) and len(dt_array) == 1):
            errors.append(f"Cadence is >= {max_cadence} seconds ({max(dt_array)/1e9} seconds)")
        if any(dt_array >= max_gap*1e9):
            errors.append(f"Data gap detected >= {max_gap/60} minutes ({max(dt_array)/1e9} seconds)")

    # Remove sites with constant values for more than 5min
    time_array = numpy.array([t.timestamp() for t in time_meas])  # convert to seconds
    window_s = 300  # 5 minutes in seconds
    # Find runs of constant values
    change_idx = numpy.where(diffs != 0)[0] + 1
    if change_idx.size:
        run_starts = numpy.concatenate(([0], change_idx))
        run_ends = numpy.concatenate((change_idx, [len(data_array)]))
    else:
        run_starts = numpy.array([0])
        run_ends = numpy.array([len(data_array)])
    for start, end in zip(run_starts, run_ends):
        if end - start > 1:
            elapsed_s = time_array[end - 1] - time_array[start]
            if elapsed_s >= window_s:
                errors.append(f"Data is constant for at least 5 minutes starting at time {time_meas[start]}")

    return errors


    