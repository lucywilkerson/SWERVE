def config(event='2024-May-Storm'):
  import os
  import datetime

  import utilrsw

  console_format = u'%(message)s'

  if event == '2024-May-Storm':
    file_path = os.path.dirname(os.path.abspath(__file__)) # Path of this script.
    data_dir = os.path.abspath(os.path.join(file_path, '..', '..', event + '-data'))
    if not os.path.exists(data_dir):
      raise FileNotFoundError(f"Data directory '{data_dir}' does not exist. Please check the path or download the data.")

    limits_data = [
      datetime.datetime(2024, 5, 10, 0, 0),
      datetime.datetime(2024, 5, 13, 0, 0)
    ]
    limits_plot = [
      datetime.datetime(2024, 5, 10, 11, 0),
      datetime.datetime(2024, 5, 12, 6, 0)
    ]

    return {
      'logger': utilrsw.logger,
      'logger_kwargs': {
        'log_dir': os.path.join(data_dir, '_log'),
        'console_format': console_format
      },
      'limits': {
        'data': limits_data, # Pad or trim data to these limits
        'plot': limits_plot  # Plot data within these limits
      },
      'dirs': {
        'data': data_dir,
        'paper': os.path.abspath(os.path.join(file_path, '..', '..', event + '-paper')),
        'processed': os.path.join(data_dir, '_processed'),
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
          'all': os.path.join(data_dir, '_processed', 'all.pkl'),
          'info': os.path.abspath(os.path.join(file_path, '..', 'info', 'info.csv')),
          'info_json': os.path.abspath(os.path.join(file_path, '..', 'info', 'info.json')),
          'info_extended': os.path.abspath(os.path.join(file_path, '..', 'info', 'info.extended.csv')),
          'info_extended_json': os.path.abspath(os.path.join(file_path, '..', 'info', 'info.extended.json')),
          'nerc_gdf': os.path.join(data_dir, 'nerc', 'nerc_gdf.geojson'),
          'shape': {
              'transmission_lines': os.path.join(data_dir, 'shape', 'Electric__Power_Transmission_Lines', 'Electric__Power_Transmission_Lines.shp'),
              'mag_lat': os.path.join(data_dir, 'shape', 'wmm_all', 'I_2024.shp')
          },
          'beta': os.path.join(data_dir, 'pulkkinen', 'waveforms_All.mat'),
      },
      'paper_sids': {
        'GIC': {
          'timeseries': {
            'Bull Run': 'a)',
            'Montgomery': 'c)',
            'Union': 'e)',
            'Widows Creek': 'g)'
          },
          'correlation': {
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
          'correlation': {
            'Bull Run': 'b)',
            '50116': 'd)'
          }
         }
      },
    }
