import numpy
import datetime

# This function will read the raw GIC data and determine if there is an error with the timeseries. If there is, it will log the error and output it
# will be added to info.py and run before metrics are calculated

def find_errors(sid, logger=None, data_type='GIC', low_signal_threshold=3, baseline_buffer=0.5, std_limit=10):
    """low_signal_threshold, baseline_buffer, and std_limit in [A]"""
    from swerve import config, site_read, cadence
    CONFIG = config()
    data = site_read(sid, data_types=data_type, logger=logger)
    data_sources = data[data_type]['measured'].keys()
    for data_source in data_sources:
        # Removing NERC sites that are TVA duplicates
        if data_source == 'NERC' and 'sid_duplicates' in CONFIG and sid in CONFIG['sid_duplicates']:
            return f"x Duplicate site ID in TVA data: mapped to '{CONFIG['sid_duplicates'][sid]}'"
        
        data_meas = data[data_type]['measured'][data_source]['original']['data']
        time_meas = data[data_type]['measured'][data_source]['original']['time']

        # Removing any sites with all negative or all positive values
        if all(i >= 0 for i in data_meas):
            return f"x All GIC values are positive for data source '{data_source}'"
        if all(i <= 0 for i in data_meas):
            return f"x All GIC values are negative for data source '{data_source}'"
        
        # Removing sites with low signal (all values within +/- low_signal_threshold A)
        if all(-low_signal_threshold <= i <= low_signal_threshold for i in data_meas):
            return f"x Low signal: all GIC values within +/- {low_signal_threshold} A for data source '{data_source}'"
        
        # Removing any sites with dt >= 1min
        dt = cadence(time_meas, logger=logger, logger_indent=2) #returns cadence in ns
        dt_array = (numpy.array(dt)).astype(numpy.float64)
        if any(dt_array >= 60e9):
            return f"x Cadence is greater than 1 minute ({max(dt_array)/1e9} seconds) for data source '{data_source}'"
        
        # Removing all sites with baseline offset
        if numpy.mean(data_meas) > baseline_buffer or numpy.mean(data_meas) < -baseline_buffer:
            return f"x Baseline offset detected (mean value: {numpy.mean(data_meas)} A) for data source '{data_source}'"

        # Removing noisy sites
        if numpy.std(data_meas) > std_limit:
            return f"x Excessive noise detected (std dev: {numpy.std(data_meas)} A) for data source '{data_source}'"
        
        # Remove sites with constant values for more than 5min
        data_array = numpy.array(data_meas).ravel()
        time_array = numpy.array([t.timestamp() for t in time_meas])  # convert to seconds
        window_s = 300  # 5 minutes in seconds
        # Find runs of constant values
        diff = numpy.diff(data_array)
        change_idx = numpy.where(diff != 0)[0] + 1
        run_starts = numpy.concatenate(([0], change_idx))
        run_ends = numpy.concatenate((change_idx, [len(data_array)]))
        for start, end in zip(run_starts, run_ends):
            if end - start > 1:
                elapsed_s = time_array[end - 1] - time_array[start]
                if elapsed_s >= window_s:
                    return f"x Data is constant for at least 5 minutes starting at time {time_meas[start]}"


    return None


    