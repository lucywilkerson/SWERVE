def config(dirs_only=False): #TODO: dirs_only option
  import os
  import datetime
  import yaml

  import utilrsw

  from swerve.cli import cli
  from swerve.util.write_event_dict import write_event_dict
  args = cli('config.py')
  if args['config'] is None:
    dirs_only = True
    #raise ValueError("No run configuration specified.")
  else:
    # Use run configuration from command line argument if provided.
    run_config_file = os.path.abspath(os.path.join('configs', args['config']))
    print(f"Using run configuration '{run_config_file}' from command line argument.")

  console_format = u'%(message)s'

  file_path = os.path.dirname(os.path.abspath(__file__)) # Path of this script.
  data_dir = os.path.abspath(os.path.join(file_path, '..', '..', f'SWERVE-data'))
  common_dir = os.path.abspath(os.path.join(file_path, '..', '..', 'SWERVE-common')) # Common data directory for all events.

  if dirs_only:
    config_dict =  {
          'dirs': {
            'data': data_dir,
            'original': os.path.join(data_dir, 'data_original'),
            'processed': os.path.join(data_dir, 'data_processed'),
          },
        }
    return config_dict

  with open(run_config_file) as f:
      conf = yaml.safe_load(f)

  configured_event = conf.get('event')
  event_request = 'all' if configured_event == 'all' or configured_event == ['all'] else None
      
  default_run_config_name = os.path.splitext(os.path.basename(args['config']))[0]
  info_dir = os.path.abspath(os.path.join(file_path, '..', 'info', conf.get('run_config_name', default_run_config_name)))
  
  # If event_dict.json exists in the corresponding info directory, use it. Otherwise, create it from the run configuration file.
  event_dict_file = os.path.abspath(os.path.join(info_dir, 'events_dict.json'))
  if os.path.exists(event_dict_file):
    import json
    with open(event_dict_file, 'r') as f:
      event_dict = json.load(f)
    # Check if *any* event in the list is missing from the dictionary
    event_list = conf.get("event", [])
    configured_events = event_list if isinstance(event_list, list) else [event_list]
    missing_event = any(str(event) not in event_dict for event in configured_events)
    if missing_event and configured_event != 'all':
      event_dict = write_event_dict(conf, file_path, data_dir)
  else:
    event_dict = write_event_dict(conf, file_path, data_dir)

  if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory '{data_dir}' does not exist. Please check the path or download the data.")

  config_dict =  {
      'event': event_dict, # Dict of event info
      'event_request': event_request,
      'logger': utilrsw.logger,
      'logger_kwargs': {
        'log_dir': os.path.join(info_dir, '_log'),
        'console_format': console_format,
        'rm_existing': False,
        'rm_empty': False
      },
      'limits': {
        'data': None, # Pad or trim data to these limits
        'plot': None  # Plot data within these limits
      },
      'info_kwargs': {
                  'data_type': conf.get('data_type', ['GIC', 'DMM', 'B']), # If specified, only return sites with this data type (e.g., GIC, B)
                  'data_source': conf.get('data_source', None), # If specified, only return sites with this data source (e.g., TVA, NERC, SWMF)
                  'data_class': conf.get('data_class', ['measured', 'calculated']), # If specified, only return sites with this data class (e.g., measured, calculated)
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
          'events_dict': os.path.join(info_dir, 'events_dict.json'),
          'info': os.path.join(info_dir, 'info.csv'),
          'info_json': os.path.join(info_dir, 'info.json'),
          'info_extended': os.path.join(info_dir, 'info.extended.csv'),
          'info_extended_json': os.path.join(info_dir, 'info.extended.json'),
          'stats_summary': os.path.join(info_dir, 'summary_table', 'stats_summary.md'),
          'config_nerc': os.path.join(file_path, 'config_nerc.json'),
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

  return config_dict
