all = [
  'add_subplot_label',
  'cadence',
  'config',
  'plt_config',
  'savefig',
  'savefig_paper',
  'subset',
  'read_info_df',
  'read_info_dict'
]

from .cadence import cadence
from .subset import subset
from .resample import resample
from .read_site import read_site
from .plot_site import plot_site

def config(event='2024-May-Storm'):
  import os
  import datetime
  from utilrsw import logger

  console_format = u'%(message)s'

  if event == '2024-May-Storm':
    data_dir = os.path.join('..', event + '-data')
    limits_data = [datetime.datetime(2024, 5, 10, 0, 0), datetime.datetime(2024, 5, 13, 0, 0)]
    limits_plot = [datetime.datetime(2024, 5, 10, 11, 0), datetime.datetime(2024, 5, 12, 6, 0)]
    return {
      'logger': logger,
      'logger_kwargs': {
        'log_dir': os.path.join(data_dir, '_log'),
        'console_format': console_format
      },
      'data_dir': data_dir,
      'paper_dir': os.path.join('..', '2024-May-Storm-paper', 'figures'),
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
      'files': {
          'mage': {
              'bcwind': os.path.join(data_dir, 'imf_data', 'bcwind.h5')
          },
          'swmf': {
            'bcwind': os.path.join(data_dir, 'imf_data', 'Dean_IMF.txt')
          },
          'cc': os.path.join(data_dir, '_results', 'cc.pkl'),
          'all': os.path.join(data_dir, '_all', 'all.pkl'),
          'info': 'info/info.csv',
          'info_extended': 'info/info.extended.csv',
          'shape_files': {
              'electric_power': os.path.join(data_dir, 'Electric__Power_Transmission_Lines', 'Electric__Power_Transmission_Lines.shp'),
              'mag_lat': os.path.join(data_dir, 'wmm_all', 'I_2024.shp')
          },
          'beta': os.path.join(data_dir, 'pulkkinen', 'waveforms_All.mat'),
      },
      'limits': {
        'data': limits_data, # Pad or trim data to these limits
        'plot': limits_plot  # Plot data within these limits
      }
    }

def format_df(df, float_fmt=".2f"):
    """
    Replace NaN with empty strings for md and tex output and format numeric
    values to two decimal places.
    """
    import numpy as np
    def format_cell(s):
        if isinstance(s, str):
            return s
        if np.isnan(s):
            return ""
        return f"{s:{float_fmt}}"

    # Apply format_cell to each cell
    return df.map(format_cell)

def read_info_dict(sid=None):
  import json
  CONFIG = config()
  info_file = CONFIG['files']['info'].replace('.csv', '.json')
  with open(info_file, 'r') as f:
    info_dict = json.load(f)

  if sid is not None:
    if sid not in info_dict:
      raise ValueError(f"sid '{sid}' not found in info.json")
    info_dict = info_dict[sid]

  return info_dict

def read_info_df(extended=False):
  import pandas
  from swerve import config
  CONFIG = config()
  file = CONFIG['files']['info_extended'] if extended else CONFIG['files']['info']
  print(f"Reading {file}")
  info_df = pandas.read_csv(file)

  # Remove rows that have errors
  info_df = info_df[~info_df['error'].str.contains('', na=False)]
  # Remove rows that don't have data_type = GIC and data_class = measured
  info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
  info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
  return info_df

def plt_config():
  import datetime
  import matplotlib.pyplot as plt

  plt.rcParams['font.family'] = 'Times New Roman'
  plt.rcParams['mathtext.fontset'] = 'cm'
  plt.rcParams['axes.titlesize'] = 18
  plt.rcParams['xtick.labelsize'] = 14
  plt.rcParams['ytick.labelsize'] = 14
  plt.rcParams['legend.fontsize'] = 14
  plt.rcParams['axes.labelsize'] = 16
  plt.rcParams['legend.fontsize'] = 14
  plt.rcParams['figure.dpi'] = 100
  plt.rcParams['savefig.dpi'] = 600

  return {
    'data': [
      datetime.datetime(2024, 5, 10, 0, 0),
      datetime.datetime(2024, 5, 13, 0, 0)
    ],
    'xlims': [
      datetime.datetime(2024, 5, 10, 11, 0),
      datetime.datetime(2024, 5, 12, 6, 0)
    ]
  }

def add_subplot_label(ax, label, loc=(-0.15, 1)):
  import matplotlib.pyplot as plt
  ax.text(*loc, label, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

def savefig(base_dir, fname, logger, logger_indent=0, root_dir=None, fmts=['png','pdf']):
  import os
  logger_indent = ' ' * logger_indent

  if root_dir is None:
    root_dir = config()['data_dir']

  from matplotlib import pyplot as plt

  base_dir = os.path.join(root_dir, base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  fname = os.path.join(base_dir, fname)

  for fmt in fmts:
    logger.info(f"{logger_indent}Writing {fname}.{fmt}")
    if fmt == 'png':
      plt.savefig(f'{fname}.{fmt}', dpi=600, bbox_inches='tight')
    else:
      plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def savefig_paper(base_dir, fname, logger, logger_indent=0):
  conf = config()
  if 'paper_dir' not in conf:
    return
  savefig(base_dir, fname, logger, logger_indent=logger_indent, root_dir=conf['paper_dir'], fmts=['pdf'])
