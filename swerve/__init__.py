from .config import config
from .subset import subset
from .cadence import cadence
from .resample import resample
from .site_read import site_read
from .site_plot import site_plot
from .site_stats import site_stats
from .site_stats_summary import site_stats_summary

def sids(sids_only=None):
  from swerve import config, read_info_dict
  CONFIG = config()

  # TODO: Add keyword exclude_errors?

  # TODO: Seems like this should be calling read_info_df() and getting all unique sids.
  info = read_info_dict()
  all_sids = list(info.keys())

  if sids_only is None:
    all_sids = [sid for sid in all_sids if not sid.startswith('test')] # Remove test sids unless keyword test is passed
    return all_sids

  # Handle keywords 'paper' and 'test'
  special_keys = {'paper': 'paper_sids', 'test': 'test_sids'}

  for sid in set(sids_only) & special_keys.keys():
    new_df = read_info_df(key=special_keys[sid])
    sids_only = list(new_df['site_id'].unique())

  # Validate sids
  for sid in sids_only:
    if sid not in info:
      raise ValueError(f"sid '{sid}' not found in info.json")

  return sids_only

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

def infodf2dict(info_df, logger):
  """
  Converts info/info.csv, which has the form

  Bull Run,36.0193,-84.1575,GIC,measured,TVA,
  Bull Run,36.0193,-84.1575,GIC,calculated,TVA,"error message"
  Bull Run,36.0193,-84.1575,GIC,calculated,MAGE,
  Bull Run,36.0193,-84.1575,GIC,calculated,GMU,
  Bull Run,36.0193,-84.1575,B,measured,TVA,
  Bull Run,36.0193,-84.1575,B,calculated,SWMF,
  Bull Run,36.0193,-84.1575,B,calculated,MAGE,

  to a dict of the form

  {
    "Bull Run": {
      "GIC": {
        "measured": "TVA",
        "calculated": ["TVA", "GMU, "MAGE"]
      },
      "B": {
        "calculated": ["SWMF", "MAGE"]
      }
    }
  }

  and saves in info/info.json
  """

  info_dict = {}

  #print(f"Preparing {CONFIG['files']['info_extended_json']}")
  for idx, row in info_df.iterrows():

    site = row['site_id']
    error = row['error']

    if isinstance(error, str) and error.startswith("x "):
      msg = f"  Skipping site '{site}' due to error message:\n    {error}"
      logger.info(msg)
      continue

    if site not in info_dict:
      info_dict[site] = {}

    data_type = row['data_type']
    if data_type not in info_dict[site]:  # e.g., GIC, B
      info_dict[site][data_type] = {}

    data_class = row['data_class']
    if data_class not in info_dict[site][data_type]:  # e.g., measured, calculated
      info_dict[site][data_type][data_class] = {}

    data_source = row['data_source']
    if data_source not in info_dict[site][data_type][data_class]:  # e.g., TVA, NERC, SWMF, OpenGGCM
      info_dict[site][data_type][data_class][data_source] = {}

    site_metadata = row.to_dict()
    for key in ['site_id', 'data_type', 'data_class', 'data_source']:
      # Remove keys that are a part of dict structure (so redundant).
      site_metadata.pop(key, None)

    info_dict[site][data_type][data_class][data_source][site] = site_metadata

  return info_dict

def read_info_dict(sid=None):
  import json
  CONFIG = config()
  info_file = CONFIG['files']['info_extended_json']
  with open(info_file, 'r') as f:
    info_dict = json.load(f)

  if sid is not None:
    if sid not in info_dict:
      raise ValueError(f"sid '{sid}' not found in {info_file}")
    info_dict = info_dict[sid]

  return info_dict

def read_info_df(extended=False, data_type=None, data_source=None, data_class=None, exclude_errors=False, key=None):
  import pandas
  from swerve import config
  CONFIG = config()
  file = CONFIG['files']['info_extended'] if extended else CONFIG['files']['info']
  print(f"    Reading {file}")
  info_df = pandas.read_csv(file)

  if key == 'paper_sids' or key == 'test_sids':
    key_sids = list(CONFIG[key]['GIC']['timeseries'])
    info_df = info_df[info_df['site_id'].isin(key_sids)]

  def filter_df(df, col, val):
    if val is None:
      return df
    if isinstance(val, list):
      return df[df[col].isin(val)]
    else:
      return df[df[col].astype(str).str.contains(str(val), na=False)]

  if exclude_errors:
    # Remove rows that have errors
    info_df = info_df[~info_df['error'].str.contains('', na=False)]

  info_df = filter_df(info_df, 'data_type', data_type)
  info_df = filter_df(info_df, 'data_source', data_source)
  info_df = filter_df(info_df, 'data_class', data_class)

  return info_df

def plt_config(scale=1):
  import matplotlib.pyplot as plt

  plt.rcParams['font.family'] = 'Times New Roman'
  plt.rcParams['mathtext.fontset'] = 'cm' # Computer Modern
  plt.rcParams['axes.titlesize'] = int(18*scale)
  plt.rcParams['axes.labelsize'] = int(16*scale)
  plt.rcParams['xtick.labelsize'] = int(14*scale)
  plt.rcParams['ytick.labelsize'] = int(14*scale)
  plt.rcParams['legend.fontsize'] = int(14*scale)
  plt.rcParams['figure.dpi'] = 200
  plt.rcParams['savefig.dpi'] = 600
  plt.rcParams['figure.constrained_layout.use'] = True

def add_subplot_label(ax, label, loc=(-0.15, 1)):
  ax.text(*loc, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

def format_cc_scatter(ax):

  # Sets the aspect ratio to make the plot square and ensure xlim and ylim are the same
  ax.set_aspect('equal', adjustable='box')

  # Set the limits to be the same for both axes
  max_x = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
  max_y = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
  max_xy = max(max_x, max_y)
  limits = [-max_xy, max_xy]

  # Set the limits for both axes
  ax.set_xlim(limits)
  ax.set_ylim(limits)

  # Draw diagonal line
  ax.plot([limits[0], limits[1]], [limits[0], limits[1]], color=3 * [0.6], linewidth=0.5)

  # Draw y = 0 line
  ax.plot([limits[0], limits[1]], [0, 0], color='black', linewidth=0.5)

  # Draw x = 0 line
  ax.plot([0, 0], [limits[0], limits[1]], color='black', linewidth=0.5)

  # Force ticks to be rendered to account for new limits and aspect ratio.
  # draw() call forces internal calculation on the tick labels, which
  # will be adjusted to have fewer ticks.
  fig = ax.get_figure()
  fig.canvas.draw()

  # Get the updated xticks, which should be fewer than the number of yticks
  # to prevent overlapping labels.
  ticks = ax.get_xticks()

  # Set the same ticks for both axes
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)

  # Re-set the limits for both axes, which may have been modified by draw()
  ax.set_xlim(limits)
  ax.set_ylim(limits)

  ax.minorticks_on()
  ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

def savefig(base_dir, fname, logger, logger_indent=0, root_dir=None, fmts=['png','pdf']):
  import os
  logger_indent = ' ' * logger_indent

  if root_dir is None:
    root_dir = config()['dirs']['data']

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
  CONFIG = config()
  if 'paper' not in CONFIG['dirs']:
    return
  savefig(base_dir, fname, logger, logger_indent=logger_indent, root_dir=CONFIG['dirs']['paper'], fmts=['pdf'])