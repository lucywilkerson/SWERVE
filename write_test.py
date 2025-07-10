import numpy as np
import pandas as pd
import random
import os
from swerve import config

CONFIG = config()

limits = CONFIG['limits']['data']
DATA_DIR = CONFIG['dirs']['data']

def write_timeseries(start_time, stop_time, value_range, data_type, data_class, nan_interval=None, seed=None, plot=False):
    """
    Writes a timeseries with given cadence (freq) from start_time to stop_time.
    Values are integers withing value_range.
    Optionally inserts NaN values at every nan_interval seconds at random positions.

    Args:
        start_time (str): Start time in 'YYYY-MM-DD HH:MM:SS' format.
        stop_time (str): Stop time in 'YYYY-MM-DD HH:MM:SS' format.
        output_file (str): Path to output CSV file.
        value_range (list): Range of integer values to generate (e.g., [-30, 30]).
        freq (str, optional): Frequency of the timeseries. Default is '1s'.
        nan_interval (int, optional): Interval (in seconds) to insert NaN values.
        seed (int, optional): Random seed for reproducibility.
        plot (bool, optional): If True, plots the generated timeseries.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Determine buffer based on data type and data class
    if data_type == 'GIC':
        if data_class == 'measured':
            freq = '1s'
            val_buffer = 1
        elif data_class == 'calculated':
            freq = '1min'
            val_buffer = 10
    elif data_type == 'B':
        if data_class == 'measured':
            freq = '1s'
            val_buffer = 5
        elif data_class == 'calculated':
            freq = '1min'
            val_buffer = 50

    # Generate time index
    times = pd.date_range(start=start_time, end=stop_time, freq=freq)
    n = len(times)

    # Generate random integer values as floats to allow NaN assignment
    # Generate values such that each next point is within +/-10 of the previous
    values = np.empty(n)
    values[0] = np.random.randint(value_range[0], value_range[1] + 1)
    for i in range(1, n):
        low = max(value_range[0], values[i-1] - val_buffer)
        high = min(value_range[1], values[i-1] + val_buffer)
        values[i] = np.random.randint(int(low), int(high) + 1)
    values = values.astype(float)

    # Optionally insert NaNs
    if nan_interval is not None and nan_interval > 0:
        nan_indices = random.sample(range(n), k=n // nan_interval)
        values[nan_indices] = np.nan

    # Create DataFrame and write to CSV
    df = pd.DataFrame({'time': times, 'value': values})

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        plt.plot(df['time'], df['value'], linestyle='-')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Generated Timeseries')
        plt.tight_layout()
        plt.grid()
        plt.show()

    output_file = os.path.join(DATA_DIR, 'test', f'test1_{data_type}_{data_class}_timeseries.csv')
    df.to_csv(output_file, index=False)

# Example usage:
if __name__ == "__main__":
    write_timeseries(
        start_time=limits[0],
        stop_time=limits[1],
        value_range=[-30, 30],
        data_type='GIC',
        data_class='measured',
        nan_interval=10,  # Set to None to disable NaNs
        seed=42
    )

    write_timeseries(
        start_time=limits[0],
        stop_time=limits[1],
        value_range=[-30, 30],
        data_type='GIC',
        data_class='calculated',
        nan_interval=10,  # Set to None to disable NaNs
        seed=42
    )

    write_timeseries(
        start_time=limits[0],
        stop_time=limits[1],
        value_range=[0, 500],
        data_type='B',
        data_class='measured',
        nan_interval=10,  # Set to None to disable NaNs
        seed=42
    )

    write_timeseries(
        start_time=limits[0],
        stop_time=limits[1],
        value_range=[0, 500],
        data_type='B',
        data_class='calculated',
        nan_interval=10,  # Set to None to disable NaNs
        seed=42
    )