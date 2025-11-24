def config():
  import os
  import datetime

  import utilrsw

  from swerve.cli import cli
  args = cli('config.py')
  if args['event'] is None:
    event = '2024-05-10'
    print(f"No event specified, defaulting to '{event}'.")
  else:
    # Use event from command line argument if provided.
    event = args['event']
    print(f"Using event '{event}' from command line argument.")

  console_format = u'%(message)s'

  file_path = os.path.dirname(os.path.abspath(__file__)) # Path of this script.
  info_dir = os.path.abspath(os.path.join(file_path, '..', 'info', event))
  data_dir = os.path.abspath(os.path.join(file_path, '..', '..', f'SWERVE-{event}'))

  common_dir = os.path.abspath(os.path.join(file_path, '..', '..', 'SWERVE-common')) # Common data directory for all events.

  if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory '{data_dir}' does not exist. Please check the path or download the data.")

  config_dict =  {
      'logger': utilrsw.logger,
      'logger_kwargs': {
        'log_dir': os.path.join(data_dir, '_log'),
        'console_format': console_format
      },
      'limits': {
        'data': None, # Pad or trim data to these limits
        'plot': None  # Plot data within these limits
      },
      'find_errors_kwargs': {'low_signal_threshold':4, # [A]
                            'baseline_buffer':1, # [A]
                            'spike_threshold':40, # [A]
                            'std_limit':15, # [A]
                            'max_cadence':60, # [s]
                            'max_gap':600 # [s]
      },
      'dirs': {
        'data': data_dir,
        'processed': os.path.join(data_dir, 'data_processed'),
      },
      'files': {
          'mage': {
              'bcwind': os.path.join(data_dir, 'imf_data', 'bcwind.h5')
          },
          'swmf': {
            'bcwind': os.path.join(data_dir, 'imf_data', 'Dean_IMF.txt')
          },
          'gmu': {
            'sim_file': os.path.join(data_dir, 'gmu', 'gic_mean_df_1.csv')
          },
          'cc': os.path.join(data_dir, '_results', 'cc.pkl'),
          'all': os.path.join(data_dir, 'data_processed', 'all.pkl'),
          'info': os.path.join(info_dir, 'info.csv'),
          'info_json': os.path.join(info_dir, 'info.json'),
          'info_extended': os.path.join(info_dir, 'info.extended.csv'),
          'info_extended_json': os.path.join(info_dir, 'info.extended.json'),
          'nerc_gdf': os.path.join(common_dir, 'nerc_gdf', 'nerc_gdf.geojson'),
          'shape': {
              'transmission_lines': os.path.join(common_dir, 'shape', 'Electric__Power_Transmission_Lines', 'Electric__Power_Transmission_Lines.shp'),
              'mag_lat': os.path.join(common_dir, 'shape', 'wmm_all', 'I_2024.shp')
          },
          'beta': os.path.join(common_dir, 'pulkkinen', 'waveforms_All.mat'),
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

  if event == '2024-05-10':

    config_dict['limits']['data'] = [
      datetime.datetime(2024, 5, 10, 15, 0),
      datetime.datetime(2024, 5, 12, 6, 0)
    ]
    config_dict['limits']['plot'] = [
      datetime.datetime(2024, 5, 10, 11, 0),
      datetime.datetime(2024, 5, 12, 6, 0)
    ]

    config_dict['nerc_prefix'] = '2024E04'
    config_dict['dirs']['paper'] = os.path.abspath(os.path.join(file_path, '..', '..', '2024-May-Storm-paper'))

    config_dict['sid_duplicates'] = {'10197':'Sullivan',
                  '10204':'Shelby',
                  '10208':'Rutherford',
                  '10203':'Raccoon Mountain',
                  '10212':'Pinhook',
                  '10201':'Montgomery',
                  '10660':'Gleason',
                  '10200':'East Point',
                  '10207':'Bull Run'
                  }

    config_dict['paper_sids'] = {
        'GIC': {
          'timeseries': {
            'Bull Run': 'a)',
            'Montgomery': 'c)',
            'Union': 'e)',
            'Widows Creek': 'g)'
          },
          'scatter': {
            'Bull Run': 'b)',
            'Montgomery': 'd)',
            'Union': 'f)',
            'Widows Creek': 'h)'
          }
        },
        'B': {
          'timeseries': {
            'Bull Run': 'a)',
            '50116': 'c)'
          },
          'scatter': {
            'Bull Run': 'b)',
            '50116': 'd)'
          }
         }
      }

  if event =='2024-10-10':

    config_dict['nerc_prefix'] = '2024E11'

    config_dict['limits']['data'] = [
      datetime.datetime(2024, 10, 10, 14, 0),
      datetime.datetime(2024, 10, 11, 14, 0)
    ]
    config_dict['limits']['plot'] = [
      datetime.datetime(2024, 10, 10, 12, 0),
      datetime.datetime(2024, 10, 11, 14, 0)
    ]

  if event =='2024-10-07':

    config_dict['nerc_prefix'] = '2024E10'

    config_dict['limits']['data'] = [
      datetime.datetime(2024, 10, 7, 12, 0),
      datetime.datetime(2024, 10, 8, 12, 0)
    ]
    config_dict['limits']['plot'] = [
      datetime.datetime(2024, 10, 7, 10, 0),
      datetime.datetime(2024, 10, 8, 12, 0)
    ]

  return config_dict
