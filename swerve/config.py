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

  file_path = os.path.dirname(os.path.abspath(__file__)) # Path of this script.
  info_dir = os.path.abspath(os.path.join(file_path, '..', 'info', conf.get('run_config_name', 'default')))
  data_dir = os.path.abspath(os.path.join(file_path, '..', '..', f'SWERVE-data'))

  common_dir = os.path.abspath(os.path.join(file_path, '..', '..', 'SWERVE-common')) # Common data directory for all events.

  # If event_dict.json exists in the corresponding info directory, use it. Otherwise, create it from the run configuration file.
  event_dict_file = os.path.abspath(os.path.join(info_dir, 'events_dict.json'))
  if os.path.exists(event_dict_file):
    import json
    with open(event_dict_file, 'r') as f:
      event_dict = json.load(f)
    # Check if *any* event in the list is missing from the dictionary
    event_list = conf.get("event", [])
    if any(event not in event_dict for event in event_list) and event_list != 'all':
      event_dict = _write_event_dict(conf, file_path)
  else:
    event_dict = _write_event_dict(conf, file_path)

  if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory '{data_dir}' does not exist. Please check the path or download the data.")

  config_dict =  {
      'event': event_dict, # Dict of event info
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

def _write_event_dict(conf, file_path):
  """
  Write event_dict to events_dict.json in the corresponding info directory.
  """
  import json
  import os
  import datetime
  from datetime import timedelta
  from swerve import storm_time

  # Getting events from run config
  events = conf.get('event', None)
  
  # If no event is specified, use start_time and stop_time from config file to create an event
  if events == None and conf.get('start_time'):
    times = [conf.get('start_time'), conf.get('stop_time')]
    events = [f"{times[0].strftime('%Y-%m-%d')}"]
  elif events == None or events == []:
    raise ValueError("No event or start_time/stop_time specified in configuration file.")

  # Finding all events with data from sources included in passed configuration file
  if events == 'all' or events == ['all']:
    import re
    data_sources = conf.get('data_source', None)
    data_dir_orig = os.path.join(file_path, '..', '..', f'SWERVE-data', 'data_original')
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    unique_events = set()
    for data_source in data_sources:
      data_source_dir = os.path.join(data_dir_orig, data_source.lower())
      if os.path.isdir(data_source_dir):
        source_events = [subdir for subdir in os.listdir(data_source_dir) if re.match(date_pattern, subdir)]
        unique_events.update(source_events)
      else:
        raise ValueError(f"Data for data_source {data_source} not found at {data_source_dir}.")
    events = list(unique_events)

  # Read in NERC events dict from config_nerc.json file
  with open(os.path.abspath(os.path.join(file_path, 'config_nerc.json')), 'r') as f:
    nerc_events = json.load(f)

  # Event dict to hold storm times
    event_dict = {event: {'data_limits': None, 'plot_limits': None, 'nerc_prefix': None} for event in events}

  for event in events:
    # Define data limits based on start_time and stop_time in config file
    if conf.get('start_time') and conf.get('stop_time'):
      times = [conf.get('start_time'), conf.get('stop_time')]
      for i, time in enumerate(times):
        if type(time) == datetime.date:
          times[i] = datetime.datetime.combine(time, datetime.time.min)
      event_dict[event]['data_limits'] = times
      
    # If no start_time and stop_time are specified and event is a NERC event, use the data_limits from nerc_events
    elif event in nerc_events:
      event_dict[event]['data_limits'] = [datetime.datetime.strptime(nerc_events[event]['data_limits'][0], '%Y-%m-%dT%H:%M'), datetime.datetime.strptime(nerc_events[event]['data_limits'][1], '%Y-%m-%dT%H:%M')]
      event_dict[event]['nerc_prefix'] = nerc_events[event]['nerc_prefix']   

    # Determine storm time if not specified    
    else:
      event_dict[event]['data_limits'] = storm_time(event)

    # Set plotting limits (subtract 2 hrs from start time to include pre-storm data)
    event_dict[event]['plot_limits'] = [
            event_dict[event]['data_limits'][0] - timedelta(hours=2),
            event_dict[event]['data_limits'][1]
          ]

  # Save event_dict as events_dict.json in corresponding info directory
  info_dir = os.path.abspath(os.path.join(file_path, '..', 'info', conf.get('run_config_name', 'default')))
  if not os.path.exists(info_dir):
    os.makedirs(info_dir)
  with open(os.path.join(info_dir, 'events_dict.json'), 'w') as f:
    json.dump(event_dict, f, indent=4, default=str)

  return event_dict
