def config():
  import os
  import datetime
  import yaml

  import utilrsw

  from swerve.cli import cli
  args = cli('config.py')
  if args['run_config'] is None:
    raise ValueError("No run configuration specified.")
  else:
    # Use run configuration from command line argument if provided.
    run_config_file = os.path.abspath(os.path.join('configs', args['run_config']))
    print(f"Using run configuration '{run_config_file}' from command line argument.")

  console_format = u'%(message)s'

  with open(run_config_file) as f:
    conf = yaml.safe_load(f)
  event = conf.get('event', None)

  file_path = os.path.dirname(os.path.abspath(__file__)) # Path of this script.
  info_dir = os.path.abspath(os.path.join(file_path, '..', 'info', conf.get('run_config_name', 'default')))
  data_dir = os.path.abspath(os.path.join(file_path, '..', '..', f'SWERVE-data'))

  common_dir = os.path.abspath(os.path.join(file_path, '..', '..', 'SWERVE-common')) # Common data directory for all events.

  if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory '{data_dir}' does not exist. Please check the path or download the data.")

  config_dict =  {
      'event': event,
      'logger': utilrsw.logger,
      'logger_kwargs': {
        'log_dir': os.path.join(info_dir, '_log'),
        'console_format': console_format
      },
      'limits': {
        'data': None, # Pad or trim data to these limits
        'plot': None  # Plot data within these limits
      },
      'info_kwargs': {
                  'data_type': conf.get('data_type', None), # If specified, only return sites with this data type (e.g., GIC, B)
                  'data_source': conf.get('data_source', None), # If specified, only return sites with this data source (e.g., TVA, NERC, SWMF)
                  'data_class': conf.get('data_class', None), # If specified, only return sites with this data class (e.g., measured, calculated)
                  'exclude_errors': conf.get('exclude_errors', False) # If True, excludes sites with known data issues (see info.csv 'manual_error' column)
                 #TODO: add error-type arg?
              },
      'main_kwargs': {'summary_table': conf.get('summary_table', False) # If True, creates a summary table of site statistics and metrics
                    },
      'filter_kwargs': {'spike_filt_type': conf.get('filter_kwargs', {}).get('spike_filt_type', 'difference'), # 'difference' or 'median' or None
                        'low_signal_threshold': conf.get('filter_kwargs', {}).get('low_signal_threshold', 0.1), # [A]
                        'baseline_buffer': conf.get('filter_kwargs', {}).get('baseline_buffer', 10), # [A]
                        'spike_threshold': conf.get('filter_kwargs', {}).get('spike_threshold', 0.5), # [A]
                        'median_window': conf.get('filter_kwargs', {}).get('median_window', 20), # [number of points]
                        'noise_threshold': conf.get('filter_kwargs', {}).get('noise_threshold', 4), # [unitless]
                        'max_cadence': conf.get('filter_kwargs', {}).get('max_cadence', 60), # [s]
                        'max_gap': conf.get('filter_kwargs', {}).get('max_gap', 600), # [s]
                        'max_const': conf.get('filter_kwargs', {}).get('max_const', 300) # [s]
      },
      'dirs': {
        'data': data_dir,
        'original': os.path.join(data_dir, 'data_original'),
        'processed': os.path.join(data_dir, 'data_processed'),
      },
      'files': {
          'mage': {
              'bcwind': os.path.join(data_dir, 'data_original', 'imf_data', 'bcwind.h5')
          },
          'swmf': {
            'bcwind': os.path.join(data_dir, 'data_original', 'imf_data', 'Dean_IMF.txt')
          },
          'gmu': {
            'sim_file': os.path.join(data_dir, 'data_original', 'gmu', 'gic_mean_df_1.csv')
          },
          'cc': os.path.join(data_dir, '_results', 'cc.pkl'),
          'all': os.path.join(data_dir, 'data_processed', 'all.pkl'),
          'info': os.path.join(info_dir, 'info.csv'),
          'info_json': os.path.join(info_dir, 'info.json'),
          'info_extended': os.path.join(info_dir, 'info.extended.csv'),
          'info_extended_json': os.path.join(info_dir, 'info.extended.json'),
          'stats_summary': os.path.join(info_dir, 'summary_table', 'stats_summary.md'),
          'nerc_gdf': os.path.join(common_dir, 'nerc_gdf', 'nerc_gdf.geojson'),
          'shape': {
              'transmission_lines': os.path.join(common_dir, 'shape', 'Electric__Power_Transmission_Lines', 'Electric__Power_Transmission_Lines.shp'),
              'mag_lat': os.path.join(common_dir, 'shape', 'wmm_all', 'I_2024.shp')
          },
          'beta': os.path.join(common_dir, 'pulkkinen', 'waveforms_All.mat'),
          'regression_results': {
              'gic_max': os.path.join(common_dir, 'regression_results', 'regression_results_gic_max.pkl'),
          },
      },
      'single_phase_sids':{
        '10358','10107', '10420', '10421', '10503', '10568'
      },
      'test_sids':{
        'GIC':{
          'timeseries':{
            'test1'
          },
          'correlation':{
            'test1'
          }
        },
        'B':{
          'timeseries':{
            'test1'
          },
          'correlation':{
            'test1'
          }
        }
      }
    }

  from swerve.nerc_events import nerc_events
  if len(event) == 1 and event[0] in nerc_events:
    from datetime import datetime
    config_dict['limits']['data'] = [datetime.strptime(nerc_events[event[0]]['data_limits'][0], '%Y-%m-%dT%H:%M'), datetime.strptime(nerc_events[event[0]]['data_limits'][1], '%Y-%m-%dT%H:%M')]
    config_dict['limits']['plot'] = [datetime.strptime(nerc_events[event[0]]['plot_limits'][0], '%Y-%m-%dT%H:%M'), datetime.strptime(nerc_events[event[0]]['plot_limits'][1], '%Y-%m-%dT%H:%M')]
    config_dict['nerc_prefix'] = nerc_events[event[0]]['nerc_prefix']

  else:
    from datetime import timedelta
    times = [conf.get('start_time'), conf.get('stop_time')]
    for i, time in enumerate(times):
      if type(time) == datetime.date:
        times[i] = datetime.datetime.combine(time, datetime.time.min)

    config_dict['limits']['data'] = times
    config_dict['limits']['plot'] = [
      times[0] - timedelta(hours=2),
      times[1]
    ]

  return config_dict
