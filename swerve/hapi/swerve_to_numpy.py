# Looks for data in SWERVE-data/data_processed between start and stop times
# Then writes available data to Numpy array of form [datetime, value]
# Given data type, data class, data source, start time, and stop time

import os
import pickle
import numpy as np
import pandas as pd

def swerve_to_numpy(data_source, site_id, data_type, data_class, modified, start, stop):
    from swerve import config
    from ..util.source_catalog import ANNUAL_SOURCES, EVENT_PARTITIONED_SOURCES
    """
    Converts SWERVE data to a Numpy array.

    Parameters:
    data_source (str): The data source (e.g., 'Hubert')
    site_id (str): The site ID
    data_type (str): The type of data (e.g., 'GIC')
    data_class (str): The class of data (e.g., 'measured')
    modified (bool): If true, returns modified data. Otherwise, returns original data.
    start (str): The start time for returned data
    stop (str): The stop time returned data

    Returns:
    np.ndarray: Numpy array of shape [n, 2] with columns [datetime, value].
    """
    # Find files within start/stop range
    CONFIG = config(dirs_only=True)
    data_dir = CONFIG['dirs']['processed']
    site_dir = site_id.lower().replace(' ', '')
    file_name = f'{data_type}_{data_class}_{data_source}.pkl'
    file_paths = []
    if data_source in ANNUAL_SOURCES:
        start_year = pd.Timestamp(start).year
        stop_year = pd.Timestamp(stop).year
        file_paths = [
            os.path.join(data_dir, 'non-event', str(year), site_dir, file_name)
            for year in range(start_year, stop_year + 1)
        ]
    elif data_source in EVENT_PARTITIONED_SOURCES:
        for event in os.listdir(data_dir):
            file_path = os.path.join(data_dir, event, 'sites', site_dir, 'data', file_name)
            if os.path.exists(file_path):
                file_paths.append(file_path)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")

    # Reading selected cached files and cropping to selected start/stop
    requested_start = pd.Timestamp(start)
    requested_stop = pd.Timestamp(stop)
    rows = []
    for file_path in file_paths:
        series = _read_series(file_path, data_source, data_type, data_class, modified)
        if series is None:
            continue
        times, values = series
        for timestamp, value in zip(pd.to_datetime(times), np.asarray(values)):
            if requested_start <= timestamp <= requested_stop:
                if np.ndim(value) == 1 and len(value) == 1:
                    value = value[0]
                rows.append((timestamp, value))

    rows.sort(key=lambda row: row[0])
    result = np.empty((len(rows), 2), dtype=object)
    if rows:
        result[:, 0] = [row[0].to_pydatetime() for row in rows]
        result[:, 1] = [row[1] for row in rows]
    print(result[:10])
    return result

def _read_series(file_path, data_source, data_type, data_class, modified):
    with open(file_path, 'rb') as file_handle:
        data = pickle.load(file_handle)

    variant = 'modified' if modified else 'original'
    try:
        if data_source in data:
            data = data[data_source]
    except (KeyError, TypeError):
        return None
    if variant in data:
        data = data[variant]
    elif modified:
        raise ValueError(f"Modified data requested but not available for {data_source}")
    if 'time' not in data or 'data' not in data:
        return None
    return data['time'], data['data']

if __name__ == "__main__":
    swerve_to_numpy('Hughes', 'Bu', 'GIC', 'measured', False, '2018-05-10', '2018-05-12')
    swerve_to_numpy('NERC', '10052', 'GIC', 'measured', True, '2024-05-10', '2024-05-12')