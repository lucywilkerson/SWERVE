import os
import csv
import numpy
import pandas
import pickle
import datetime

# This function will read the raw GIC data and determine if there is an error with the timeseries. If there is, it will log the error and output it
# will be added to info.py and run before metrics are calculated

def find_errors(sid, logger=None, data_type='GIC'):
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
        
        # Removing sites with constant values for more than 2min
        data_array = numpy.array(data_meas)
        window_ns = 120e9  # 2 minutes in nanoseconds
        start_idx = 0
        while start_idx < len(data_array):
            val = data_array[start_idx]
            end_idx = start_idx
            elapsed_ns = 0
            while end_idx + 1 < len(data_array) and data_array[end_idx + 1] == val:
                # Calculate time difference using time_meas
                if end_idx + 1 < len(time_meas):
                    delta = time_meas[end_idx + 1] - time_meas[end_idx]
                    elapsed_ns += delta.total_seconds() * 1e9
                else:
                    break
            if elapsed_ns >= window_ns:
                return f"x Data is constant for at least 2 minutes starting at index {start_idx} for data source '{data_source}'"
            end_idx += 1
            start_idx = end_idx + 1


    return dt


    