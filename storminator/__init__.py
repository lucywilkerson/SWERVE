import os

all = ['plt_config', 'savefig', 'savefig_paper', 'subset']

import matplotlib.pyplot as plt

LOG_DIR = 'log'
DATA_DIR = os.path.join('..', '2024-May-Storm-data')
base_dir = os.path.join(DATA_DIR, '_processed')
paper_dir = os.path.join('..','2024-May-Storm-paper')

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

def savefig(sid, fname, logger, sub_dir="", fmts=['png','pdf']):
  fdir = os.path.join(base_dir, sid.lower().replace(' ', ''), sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    logger.info(f"Writing {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def savefig_paper(fname, logger, sub_dir="", fmts=['png','pdf']):
  fdir = os.path.join(paper_dir, sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    logger.info(f"Writing {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

