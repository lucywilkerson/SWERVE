# Execute using
#   python -m swerve.filter_test

run_tests   = True  # Run tests
write_tests = True # Write test timeseries 

from datetime import timedelta
import json
from swerve import config
CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
filter_kwargs = CONFIG['filter_kwargs']

def _test_dict():
    # Returns dictionary with all test configuration information

    from swerve import config
    CONFIG = config()
    limits = CONFIG['limits']['data']

    return {
                'test_spike':{
                    'GIC':{
                        'description':{'Rand walk with max/min of +/-5, spikes every 3 hours.'},
                        'config':{
                            'start_time':limits[0],
                            'stop_time':limits[1],
                            'value_range':[-5, 5],
                            'mode':'rand',
                            'add_spikes':True
                        }
                    }
                },
                'test_offset':{
                    'GIC':{
                        'description':{'Sin wave with amplitude of 30, offset by 10.'},
                        'config':{
                            'start_time':limits[0],
                            'stop_time':limits[1],
                            'value_range':[-40, 20],
                            'mode':'sin'
                        }
                    }
                },
                'test_both':{
                    'GIC':{
                        'description':{'Rand walk with max/min of +/-10, spikes every 3 hours, offset by 5.'},
                        'config':{
                            'start_time':limits[0],
                            'stop_time':limits[1],
                            'value_range':[-15, 5],
                            'mode':'sin',
                            'add_spikes':True
                        }
                    }
                },
                'test_low_val':{
                    'GIC':{
                        'description':{'Rand walk with max/min of +/-3.9, offset by 5.'},
                        'config':{
                            'start_time':limits[0]-timedelta(hours=3),
                            'stop_time':limits[1],
                            'value_range':[-3.9-5, 3.9-5],
                            'mode':'rand',
                            'add_spikes':False
                        }
                    }
                },
                'test_long_break':{
                    'GIC':{
                        'description':{'Rand walk with max/min of +/-15, breaks every 3 hours.'},
                        'config':{
                            'start_time':limits[0],
                            'stop_time':limits[1],
                            'value_range':[-15, 15],
                            'mode':'rand',
                            'add_breaks':True
                        }
                    }
                },
                'test_cadence':{
                    'GIC':{
                        'description':{'Rand walk with max/min of +/-15, 10 min cadence.'},
                        'config':{
                            'start_time':limits[0],
                            'stop_time':limits[1],
                            'value_range':[-15, 15],
                            'mode':'rand',
                            'cadence':600
                        }
                    }
                },
            }


def _write_timeseries(test_name, start_time, stop_time, value_range, data_type, mode='sin', logger=logger,nan_interval=None, cadence=None, add_spikes=False, add_breaks=False, seed=None, plot=False):
    import os
    import random

    import numpy as np
    import pandas as pd

    from swerve import config
    CONFIG = config()
    DATA_DIR = CONFIG['dirs']['original']

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
        cadence (int, optional): Cadence in seconds for the timeseries. If None, defaults to 1 second.
        add_spikes (bool, optional): If True, adds spikes to the timeseries.
        add_breaks (bool, optional): If True, adds breaks (NaN values) to the timeseries.
        seed (int, optional): Random seed for reproducibility.
        plot (bool, optional): If True, plots the generated timeseries.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Determine buffer and frequency
    if cadence is None:
        freq = '1s'
    else:
        freq = f"{cadence}s"
    val_buffer = 1

    # Generate time index
    times = pd.date_range(start=start_time, end=stop_time, freq=freq)
    n = len(times)

    if mode == 'sin':
        # Sine wave: two full cycles over the time range
        amplitude = (value_range[1] - value_range[0]) / 2
        offset = (value_range[1] + value_range[0]) / 2
        x = np.linspace(0, 4 * np.pi, n) 
        values = amplitude * np.sin(10*x) + offset
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
    
    if add_spikes:
        spike_interval = 3 * 60 * 60 # every 3 hours
        spike_indices = np.arange(0, n, spike_interval)
        # For each spike index, add +i/ +5i / +i for even indices, -i/ -5i / -i for odd
        for i, idx in enumerate(spike_indices):
            sign = 1 if (i % 2) == 0 else -1
            values[idx-1] += sign * i
            values[idx] += sign * 5 * i
            if idx + 1 < n:
                values[idx+1] += sign * i

    if add_breaks:
        break_interval = 3 * 60 * 60 # every 3 hours
        break_indices = np.arange(0, n, break_interval)
        i = 1
        for idx in break_indices:
            break_length = i * 60 # idx minutes
            start_break = idx
            print(f"Break length: {break_length/60} min")
            end_break = min(idx + break_length, n)
            values[start_break:end_break] = np.nan
            i += 1

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
    if not os.path.exists(os.path.join(DATA_DIR, 'test')):
        os.makedirs(os.path.join(DATA_DIR, 'test'))
    output_file = os.path.join(DATA_DIR, 'test', f'{test_name}_{data_type}_{data_class}_timeseries.csv')
    df.to_csv(output_file, index=False)

    # Add test data to info.csv if not already included
    from swerve import sids
    test_sids = sids(key=['test'], logger=logger)
    if test_name not in test_sids:
        logger.info(f"Adding {test_name} to info.csv")
        new_row = {
            'site_id': test_name,
            'data_type': data_type,
            'data_class': data_class,
            'data_source': 'TEST'
        }
        from swerve import read_info_df, infodf2dict
        info_df = read_info_df(extended=False, logger=logger)
        info_df = (pd.concat([info_df, pd.DataFrame([new_row])], ignore_index=True)
                   if not info_df.empty else pd.DataFrame([new_row]))
        info_df.to_csv(CONFIG['files']['info'], index=False)
        info_extended = read_info_df(extended=True, logger=logger)
        info_extended = (pd.concat([info_extended, pd.DataFrame([new_row])], ignore_index=True)
                         if not info_extended.empty else pd.DataFrame([new_row]))
        info_extended.to_csv(CONFIG['files']['info_extended'], index=False)
        info_dict = infodf2dict(info_extended, logger)
        logger.info(f"Writing {CONFIG['files']['info_extended_json']}")
        with open(CONFIG['files']['info_extended_json'], 'w') as f:
            json.dump(info_dict, f, indent=2)
        logger.info(f"Added {test_name} to info.csv, info.extended.csv, info.extended.json.")

    # Make calculated data by averaging measured data into 1-min intervals
    df_resampled = df.copy()
    df_resampled.set_index('time', inplace=True)
    df_resampled = df_resampled.resample('1min').mean().reset_index()

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


def _test_site(site, data_types=None, plot=False, logger=logger):
    """
    Tests the filtering method and plots the original and filtered data for visual inspection.
    Args:
        site (str): The site id to test (e.g., 'test1').
        data_types (list, optional): List of data types to test (e.g., ['GIC', 'B']). Defaults to ['GIC', 'B'].
        plot (bool, optional): If True, plots the original and filtered data for visual inspection. Defaults to False.
    """

    from swerve import site_read, filter

    def test_data(data, data_type):
        # Test filtering
        logger.info(f"Testing filter for {site} {data_type}...")
        filtered_data, errors, corrections = filter(data, logger=logger)
        return filtered_data, errors, corrections

    if 'GIC' in data_types:
        # Read and parse data or use cached data if found and reparse is False.
        gic_data = site_read(site, data_types='GIC', logger=logger, reparse=True)
        # Get original test GIC data 
        orig_data = gic_data['GIC']['measured']['TEST']['original']
        filtered_data, errors, corrections = test_data(orig_data, 'GIC')    
    
    # Plot filtered data and original data for visual inspection
    if plot:
        import matplotlib.pyplot as plt
        import os
        fig_dir = os.path.join(CONFIG['dirs']['original'], 'test', 'figures')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.figure(figsize=(12, 4))
        plt.plot(orig_data['time'], orig_data['data'], color='salmon', linestyle='-', label='Original')
        plt.plot(filtered_data['time'], filtered_data['data'], color='maroon', linestyle='-', label='Filtered')
        plt.xlabel('Time')
        plt.ylabel('GIC (A)')
        plt.title(f"{site} GIC Timeseries\nErrors: {errors}\nCorrections: {corrections}")
        plt.tight_layout()
        plt.grid()
        plt.legend()
        logger.info(f"Saving plot to {os.path.join(fig_dir, f'{site}_GIC_filtered.png')}")
        plt.savefig(os.path.join(fig_dir, f"{site}_GIC_filtered.png"))
        #plt.show()


if __name__ == "__main__":
    if write_tests:
        test_dict = _test_dict()
        for test_name, test_info in test_dict.items():
            for data_type, data_info in test_info.items():
                _write_timeseries(test_name, **data_info['config'], data_type=data_type, seed=42, plot=True)
    if run_tests:
        test_dict = _test_dict()
        for test_name, test_info in test_dict.items():
            _test_site(test_name, data_types=test_info.keys(), plot=True, logger=logger)