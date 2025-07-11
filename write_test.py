import numpy as np
import pandas as pd
import random
import os
from swerve import config

CONFIG = config()

limits = CONFIG['limits']['data']
DATA_DIR = CONFIG['dirs']['data']
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

write_tests = False #Write test timeseries
test_cc_analysis = True #Run test cross-correlation analysis

def write_timeseries(start_time, stop_time, value_range, data_type, mode='sin', nan_interval=None, seed=None, plot=False):
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
    if data_type == 'GIC':
        df = pd.DataFrame({'time': times, 'value': values})
    elif data_type == 'B': #TODO: make B measured and B calculated better reflective of the data
        df = pd.DataFrame({
            'time': times,
            'valuex': values,
            'valuey': values,
            'valuez': values
        })

    # Write measured data to CSV
    data_class = 'measured'
    output_file = os.path.join(DATA_DIR, 'test', f'test1_{data_type}_{data_class}_timeseries.csv')
    df.to_csv(output_file, index=False)

    # Make calculated data by averaging measured data into 1-min intervals
    df_resampled = df.copy()
    df_resampled.set_index('time', inplace=True)
    df_resampled = df_resampled.resample('1min').mean().reset_index()
    data_class = 'calculated'
    output_file = os.path.join(DATA_DIR, 'test', f'test1_{data_type}_{data_class}_timeseries.csv')
    df_resampled.to_csv(output_file, index=False)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        if data_type == 'GIC':
            plt.plot(df['time'], df['value'], linestyle='-')
            plt.plot(df_resampled['time'], df_resampled['value'], linestyle='-')
        elif data_type == 'B':
            plt.plot(df['time'], df['valuex'], linestyle='-')
            plt.plot(df_resampled['time'], df_resampled['valuex'], linestyle='-')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.grid()
        plt.legend(['Measured', 'Calculated'])
        plt.show()

def test_cc(sites=['test1'], data_types=None, expected_cc=1.0):
    import scipy.stats
    from swerve import site_read, site_stats

    if data_types is None:
        data_types = ['GIC', 'B']

    for sid in sites:
        if 'GIC' in data_types:
            # Read and parse data or use cached data if found and reparse is False.
            data = site_read(sid, data_types='GIC', logger=logger, reparse=True)

            # Add statistics to data in data[sid].
            stats = site_stats(sid, data, data_types='GIC', logger=logger)
            cc_gic = stats['GIC/calculated/NA']['metrics']['cc'][0]
            assert cc_gic > expected_cc, f"GIC measured/calculated correlation {cc_gic} is less than expected {expected_cc}"

        if 'B' in data_types:
            # Read and parse data or use cached data if found and reparse is False.
            data = site_read(sid, data_types='B', logger=logger, reparse=True)

            # Add statistics to data in data[sid].
            stats = site_stats(sid, data, data_types='B', logger=logger)

            cc_bx = stats['B/calculated/NA']['metrics']['cc'][0]
            assert cc_bx > expected_cc, f"B x measured/calculated correlation {cc_bx} is less than expected {expected_cc}"

# Example usage:
if write_tests:
    write_timeseries(
        start_time=limits[0],
        stop_time=limits[1],
        value_range=[-30, 30],
        data_type='GIC'
    )

    write_timeseries(
        start_time=limits[0],
        stop_time=limits[1],
        value_range=[-250, 250],
        data_type='B'
    )

if test_cc_analysis:
    test_cc(data_types=['GIC'])