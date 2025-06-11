import os

all = ['plt_config', 'savefig', 'savefig_paper', 'subset', 'FILES']

import matplotlib.pyplot as plt

LOG_DIR = 'log'
DATA_DIR = os.path.join('..', '2024-May-Storm-data')
base_dir = os.path.join(DATA_DIR, '_processed')
results_dir = os.path.join(DATA_DIR, '_results')
paper_dir = os.path.join('..','2024-May-Storm-paper')

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
            {'cc': os.path.join(results_dir, 'cc.pkl')
             },
          'info':
            {'csv': os.path.join('info', 'info.csv'),
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
  plt.rcParams['xtick.labelsize'] = 14
  plt.rcParams['ytick.labelsize'] = 14
  plt.rcParams['axes.labelsize'] = 16
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

def savefig(base_dir, fname, logger, sid=None, sub_dir="", fmts=['png','pdf']):

  base_dir = os.path.join(DATA_DIR, base_dir)
  if sid is not None:
    base_dir = os.path.join(base_dir, sid.lower().replace(' ', ''), sub_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  fname = os.path.join(base_dir, fname)

  for fmt in fmts:
    logger.info(f"Writing {fname}.{fmt}")
    if fmt == 'png':
      plt.savefig(f'{fname}.{fmt}', dpi=600, bbox_inches='tight')
    else:
      plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def savefig_paper(fname, logger, sub_dir="", fmts=['png','pdf']):
  fdir = os.path.join(paper_dir, sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    logger.info(f"Writing {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def add_subplot_label(ax, label, loc=(-0.15, 1)):
  ax.text(*loc, label, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

