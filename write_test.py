import numpy as np
import pandas as pd
import random
import os
from swerve import config

CONFIG = config()

limits = CONFIG['limits']['data']
DATA_DIR = CONFIG['dirs']['data']

write_gic = True #Write GIC test timeseries
write_b = False #Write B test timeseries

def write_timeseries(start_time, stop_time, value_range, data_type, mode='sin', nan_interval=None, seed=None, plot=True):
    """
    Writes a timeseries (random or sine) from start_time to stop_time.
    Values are within value_range. Optionally inserts NaN values at every nan_interval seconds at random positions.

    Args:
        start_time (str): Start time in 'YYYY-MM-DD HH:MM:SS' format.
        stop_time (str): Stop time in 'YYYY-MM-DD HH:MM:SS' format.
        value_range (list): Range of values to generate (e.g., [-30, 30]).
        data_type (str): 'GIC' or 'B'.
        data_class (str): 'measured' or 'calculated'.
        mode (str): 'sin' for sine wave, 'rand' for random walk.
        nan_interval (int, optional): Interval (in seconds) to insert NaN values.
        seed (int, optional): Random seed for reproducibility.
        plot (bool, optional): If True, plots the generated timeseries.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Determine buffer and frequency based on data type
    freq = '1s'
    if data_type == 'GIC':
        val_buffer = 1
    elif data_type == 'B':
        val_buffer = 5

    # Generate time index
    times = pd.date_range(start=start_time, end=stop_time, freq=freq)
    n = len(times)

    if mode == 'sin':
        # Sine wave: two full cycles over the time range
        amplitude = (value_range[1] - value_range[0]) / 2
        offset = (value_range[1] + value_range[0]) / 2
        x = np.linspace(0, 4 * np.pi, n) 
        values = amplitude * np.sin(x) + offset
    elif mode == 'rand':
        # Random walk: each next point is within +/- val_buffer of the previous
        values = np.empty(n)
        values[0] = np.random.randint(value_range[0], value_range[1] + 1)
        for i in range(1, n):
            low = max(value_range[0], values[i-1] - val_buffer)
            high = min(value_range[1], values[i-1] + val_buffer)
            values[i] = np.random.randint(int(low), int(high) + 1)
        values = values.astype(float)
    else:
        raise ValueError("mode must be 'sin' or 'rand'")

    # Optionally insert NaNs
    if nan_interval is not None and nan_interval > 0:
        nan_indices = random.sample(range(n), k=n // nan_interval)
        values[nan_indices] = np.nan

    # Create DataFrame
    df = pd.DataFrame({'time': times, 'value': values})

    # Write measured data to CSV
    data_class = 'measured'
    output_file = os.path.join(DATA_DIR, 'test', f'test1_{data_type}_{data_class}_timeseries.csv')
    df.to_csv(output_file, index=False)

    # Make calculated data by averaging measured data into 1-min intervals
    df_resampled = df.copy()
    df_resampled.set_index('time', inplace=True)
    df_resampled = df_resampled.resample('1T').mean().reset_index()
    data_class = 'calculated'
    output_file = os.path.join(DATA_DIR, 'test', f'test1_{data_type}_{data_class}_timeseries.csv')
    df_resampled.to_csv(output_file, index=False)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(df['time'], df['value'], linestyle='-')
        plt.plot(df_resampled['time'], df_resampled['value'], linestyle='-')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.grid()
        plt.legend(['Measured', 'Calculated'])
        plt.show()

# Example usage:
if write_gic:
    write_timeseries(
        start_time=limits[0],
        stop_time=limits[1],
        value_range=[-30, 30],
        data_type='GIC',
        nan_interval=10,  # Set to None to disable NaNs
        seed=42
    )


if write_b:
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