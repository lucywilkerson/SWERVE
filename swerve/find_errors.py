import os
import csv
import numpy
import pandas
import pickle
import datetime

# This function will read the raw GIC data and determine if there is an error with the timeseries. If there is, it will log the error and output it
# will be added to info.py and run before metrics are calculated

def find_errors(sid, logger=None, data_type='GIC', baseline_buffer=0.5):
    """baseline_buffer in [A]"""
    from swerve import site_read, cadence
    data = site_read(sid, data_types=data_type, logger=logger)
    data_sources = data[data_type]['measured'].keys()
    for data_source in data_sources:
        data_meas = data[data_type]['measured'][data_source]['original']['data']
        time_meas = data[data_type]['measured'][data_source]['original']['time']

        # Removing any sites with all negative or all positive values
        if all(i >= 0 for i in data_meas):
            return f"x All GIC values are positive for data source '{data_source}'"
        if all(i <= 0 for i in data_meas):
            return f"x All GIC values are negative for data source '{data_source}'"
        
        # Removing any sites with dt >= 1min
        dt = cadence(time_meas, logger=logger, logger_indent=2) #returns cadence in ns
        dt_array = (numpy.array(dt)).astype(numpy.float64)
        if any(dt_array >= 60e9):
            return f"x Cadence is greater than 1 minute ({max(dt_array)/1e9} seconds) for data source '{data_source}'"
        
        # Removing all sites with baseline offset
        if numpy.mean(data_meas) > baseline_buffer or numpy.mean(data_meas) < -baseline_buffer:
            return f"x Baseline offset detected (mean value: {numpy.mean(data_meas)} A) for data source '{data_source}'"

        
        # Remove sites with constant values for more than 2min
        data_array = numpy.array(data_meas)
        time_array = numpy.array([t.timestamp() for t in time_meas])  # convert to seconds
        window_s = 120  # 2 minutes in seconds
        # Find runs of constant values
        diff = numpy.diff(data_array)
        change_idx = numpy.where(diff != 0)[0] + 1
        run_starts = numpy.concatenate(([0], change_idx))
        run_ends = numpy.concatenate((change_idx, [len(data_array)]))
        for start, end in zip(run_starts, run_ends):
            if end - start > 1:
                elapsed_s = time_array[end - 1] - time_array[start]
            if elapsed_s >= window_s:
                return f"x Data is constant for at least 2 minutes starting at time {time_meas[start]}"


    return dt


    