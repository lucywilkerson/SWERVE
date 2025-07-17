import numpy as np
import pandas as pd
import random
import os
from swerve import config

CONFIG = config()

limits = CONFIG['limits']['data']
DATA_DIR = CONFIG['dirs']['data']
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

write_tests = True #Write test timeseries
run_tests = True #Run tests

def write_timeseries(test_name, start_time, stop_time, value_range, data_type, mode='sin', nan_interval=None, seed=None, plot=False):
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
    logger.info(f"  Writing {test_name}_{data_type}_{data_class}_timeseries.csv")
    output_file = os.path.join(DATA_DIR, 'test', f'{test_name}_{data_type}_{data_class}_timeseries.csv')
    df.to_csv(output_file, index=False)

    # Make calculated data by averaging measured data into 1-min intervals
    if test_name == 'test2' and data_type == 'GIC':
        df['value'] = -df['value']
    if test_name == 'test2' and data_type == 'B':
        df['valuex'] = -values
        df['valuey'] = -values
        df['valuez'] = -values
    df_resampled = df.copy()
    df_resampled.set_index('time', inplace=True)
    df_resampled = df_resampled.resample('1min').mean().reset_index()
    data_class = 'calculated'
    logger.info(f"  Writing {test_name}_{data_type}_{data_class}_timeseries.csv")
    output_file = os.path.join(DATA_DIR, 'test', f'{test_name}_{data_type}_{data_class}_timeseries.csv')
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

def test_site(site, metrics, stats, data_types=None):
    """
    Tests the statistics and metrics for a given site using provided expected values.
    Args:
        site (str): The site id to test (e.g., 'test1').
        metrics (dict): Dictionary containing expected metric values for measured vs calculated comparison.
        stats (dict): Dictionary containing expected statistics for data statistics comparison.
        data_types (list, optional): List of data types to test (e.g., ['GIC', 'B']). Defaults to ['GIC', 'B'].
    Raises:
        AssertionError: If the calculated correlation coefficients do not match the expected values.
    Notes:
        - The function reads and parses data for the specified site and data types.
        - It computes statistics and compares the calculated correlation coefficients with the expected values from the provided dictionaries.
    """
    from swerve import site_read, site_stats

    def test_data(data_stats, metrics, stats, data_type):
        # Extract stats and metrics, check lengths, then perform tests
        data_metrics = data_stats[f'{data_type}/calculated/TEST']['metrics']
        for val in metrics.keys():
            test_stat = data_metrics[val][0]
            expected_stat = metrics[val]
            assert test_stat == expected_stat, f"{data_type} measured/calculated {val} {test_stat} is not equal to expected {expected_stat}"
        data_meas_stats = data_stats[f'{data_type}/measured/TEST']['stats']
        #data_calc_stats = data_stats[f'{data_type}/calculated/TEST']['stats']
        for val in stats.keys():
            test_stat = data_meas_stats[val]
            expected_stat = stats[val]
            tolerance = 0.001 #set tolerance for test
            assert expected_stat-tolerance <= test_stat <= expected_stat+tolerance, f"{data_type} measured {val} {test_stat} is not equal to expected {expected_stat}"

    if data_types is None:
        data_types = ['GIC', 'B']

    if 'GIC' in data_types:
        # Read and parse data or use cached data if found and reparse is False.
        gic_data = site_read(site, data_types='GIC', logger=logger, reparse=True)

        # Add statistics to data in data[sid].
        gic_stats = site_stats(site, gic_data, data_types='GIC', logger=logger) 
        
        test_data(gic_stats, metrics, stats, 'GIC')    

    if 'B' in data_types:
        # Read and parse data or use cached data if found and reparse is False.
        b_data = site_read(site, data_types='B', logger=logger, reparse=True)

        # Add statistics to data in data[sid].
        b_stats = site_stats(site, b_data, data_types='B', logger=logger)

        test_data(b_stats, metrics, stats, 'B')


# Define dictionary with all test information
test_dict = {'test1':{
                    'GIC':{
                        'description':{'Sin wave with max/min of +/-30. Measured is resampled from calculated, so pe and cc = 1.0.'},
                        'config':{
                            'start_time':limits[0],
                            'stop_time':limits[1],
                            'value_range':[-30, 30],
                        },
                        'metrics':{
                            'cc':1.0,
                            'pe':1.0
                        },
                        'stats':{
                            'max':30,
                            'min':-30
                        }
                    },
                    'B':{
                        'description':{'Sin wave with max/min of +/-250. Measured is resampled from calculated, so pe and cc = 1.0.'},
                        'config':{
                            'start_time':limits[0],
                            'stop_time':limits[1],
                            'value_range':[-250, 250],
                        },
                        'metrics':{
                            'cc':1.0,
                            'pe':1.0
                        },
                        'stats':{
                            'max':250,
                            'min':-250
                        }
                    }
                },
                'test2':{
                    'GIC':{
                        'description':{'Sin wave with max/min of +/-30. Measured is resampled from -calculated, so cc = -1.0.'},
                        'config':{
                            'start_time':limits[0],
                            'stop_time':limits[1],
                            'value_range':[-30, 30],
                        },
                        'metrics':{
                            'cc':-1.0
                        },
                        'stats':{
                            'max':30,
                            'min':-30
                        }
                    },
                    'B':{
                        'description':{'Sin wave with max/min of +/-250. Measured is resampled from -calculated, so cc = -1.0.'},
                        'config':{
                            'start_time':limits[0],
                            'stop_time':limits[1],
                            'value_range':[-250, 250],
                        },
                        'metrics':{
                            'cc':-1.0,
                        },
                        'stats':{
                            'max':250,
                            'min':-250
                        }
                    }
                }
                }



for test in test_dict.keys():
    for data_type in test_dict[test].keys():
        if write_tests:
            timeseries_config = test_dict[test][data_type]['config']
            write_timeseries(
                test_name = test,
                start_time = timeseries_config['start_time'],
                stop_time = timeseries_config['stop_time'],
                value_range = timeseries_config['value_range'],
                data_type = data_type
            )
        
        # Before running tests (for new series), add test to info.csv, and run 'python info.py' and 'python main.py test'
        if run_tests:
            test_metrics = test_dict[test][data_type]['metrics']
            test_stats = test_dict[test][data_type]['stats']
            test_site(site=test, metrics=test_metrics, stats=test_stats, data_types=data_type)