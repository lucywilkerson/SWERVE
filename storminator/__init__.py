import os

all = ['plt_config', 'savefig', 'savefig_paper', 'subset', 'add_subplot_label', 'FILES']

import matplotlib.pyplot as plt

DATA_DIR = os.path.join('..', '2024-May-Storm-data')
PAPER_DIR = os.path.join('..','2024-May-Storm-paper', 'figures')
LOG_DIR = 'log'
LOG_CFG = {'dir': LOG_DIR, 'kwargs': {'console_format': u'%(message)s'}}

FILES = {
          'mage':
            {
              'bcwind': os.path.join(DATA_DIR, 'imf_data', 'bcwind.h5')
          },
          'shape_files':
            {
              'electric_power': os.path.join(DATA_DIR, 'Electric__Power_Transmission_Lines', 'Electric__Power_Transmission_Lines.shp'),
              'geo_mag': os.path.join(DATA_DIR, 'wmm_all', 'I_2024.shp')
            },
          'analysis':
            {
              'cc': os.path.join(DATA_DIR, '_results', 'cc.pkl'),
              'beta': os.path.join(DATA_DIR, 'pulkkinen', 'waveforms_All.mat')
            },
          'info':
            {
              'csv': os.path.join('info', 'info.csv'),
              'json': os.path.join('info', 'info.json'),
              'extended': os.path.join('info', 'info.extended.csv')
              }
        }

def subset(time, data, start, stop):
  import numpy as np
  idx = np.logical_and(time >= start, time <= stop)
  if data.ndim == 1:
    return time[idx], data[idx]
  return time[idx], data[idx,:]

def plt_config():
  import datetime
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

def savefig(base_dir, fname, logger, root_dir=DATA_DIR, fmts=['png','pdf']):

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
  ax.text(*loc, label, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top', ha='left')