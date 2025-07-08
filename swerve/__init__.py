import os

all = [
  'logger',
  'subset',
  'add_subplot_label',
  'plt_config',
  'savefig',
  'savefig_paper',
  'subset',
  'read_info',
  'FILES',
  'DATA_DIR'
]

DATA_DIR = os.path.join('..', '2024-May-Storm-data')
PAPER_DIR = os.path.join('..','2024-May-Storm-paper', 'figures')

LOG_KWARGS = {
  'log_dir': 'log',
  'console_format': u'%(message)s'
}
from utilrsw import logger

FILES = {
          'mage': {
              'bcwind': os.path.join(DATA_DIR, 'imf_data', 'bcwind.h5')
          },
          'swmf': {
            'bcwind': os.path.join(DATA_DIR, 'imf_data', 'Dean_IMF.txt')
          },
          'cc': os.path.join(DATA_DIR, '_results', 'cc.pkl'),
          'all': os.path.join(DATA_DIR, '_all', 'all.pkl'),
          'info': 'info/info.csv',
          'info_extended': 'info/info.extended.csv',
          'shape_files': {
              'electric_power': os.path.join(DATA_DIR, 'shape', 'Electric__Power_Transmission_Lines', 'Electric__Power_Transmission_Lines.shp'),
              'mag_lat': os.path.join(DATA_DIR, 'shape', 'wmm_all', 'I_2024.shp')
          },
          'beta': os.path.join(DATA_DIR, 'pulkkinen', 'waveforms_All.mat'),
        }

def subset(time, data, start, stop):
  import numpy as np
  idx = np.logical_and(time >= start, time <= stop)
  if data.ndim == 1:
    return time[idx], data[idx]
  return time[idx], data[idx,:]

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
      datetime.datetime(2024, 5, 10, 15, 0),
      datetime.datetime(2024, 5, 12, 6, 0)
    ],
    'xlims': [
      datetime.datetime(2024, 5, 10, 11, 0),
      datetime.datetime(2024, 5, 12, 6, 0)
    ]
  }

def add_subplot_label(ax, label, loc=(-0.15, 1)):
  import matplotlib.pyplot as plt
  ax.text(*loc, label, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

def read_info(extended=False, data_type='GIC'):
  import pandas
  #reading in info.csv
  file = FILES['info_extended'] if extended else FILES['info']
  print(f"Reading {file}")
  info_df = pandas.read_csv(file)
  # Remove rows that have errors
  info_df = info_df[~info_df['error'].str.contains('', na=False)]
  # Remove rows that don't have data_type = GIC and data_class = measured
  if data_type == 'GIC':
    info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
  elif data_type == 'B':
    info_df = info_df[info_df['data_type'].str.contains('B', na=False)]
  info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
  return info_df

def savefig(base_dir, fname, logger, root_dir=DATA_DIR, fmts=['png','pdf']):
  from matplotlib import pyplot as plt
  base_dir = os.path.join(root_dir, base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  fname = os.path.join(base_dir, fname)

  for fmt in fmts:
    logger.info(f"Writing {fname}.{fmt}")
    if fmt == 'png':
      plt.savefig(f'{fname}.{fmt}', dpi=600, bbox_inches='tight')
    else:
      plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def savefig_paper(base_dir, fname, logger):
  savefig(base_dir, fname, logger, root_dir=PAPER_DIR, fmts=['pdf'])

def add_subplot_label(ax, label, loc=(-0.15, 1)):
  import matplotlib.pyplot as plt
  ax.text(*loc, label, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top', ha='left')