from .cli import cli
from .config import config
from .subset import subset
from .cadence import cadence
from .resample import resample
from .site_read import site_read
from .site_plot import site_plot
from .site_stats import site_stats
from .site_stats_summary import site_stats_summary
from .find_errors import find_errors
from .update_info_extended import update_info_extended

def sids(extended=True, data_type=None, data_source=None, data_class=None, exclude_errors=False, error_type='manual_error', key=None, logger=None):
  from swerve import config, read_info_df

  # Handle keywords 'paper' and 'test'
  special_keys = {'paper': 'paper_sids', 'test': 'test_sids'}

  if key is None:
    info = read_info_df(extended=extended, data_type=data_type, data_source=data_source, data_class=data_class, exclude_errors=exclude_errors, error_type=error_type, logger=logger)
  elif key[0] in list(special_keys.keys()):
    site_key = special_keys[key[0]]
    info = read_info_df(extended=extended, data_type=data_type, data_source=data_source, data_class=data_class, exclude_errors=exclude_errors,  error_type=error_type, key=site_key, logger=logger)
  else:
    info = read_info_df(extended=extended, data_type=data_type, data_source=data_source, data_class=data_class, exclude_errors=exclude_errors, error_type=error_type, logger=logger)
    info = info[info['site_id'].isin(key)]
    if info.empty and data_type is not None:
      raise ValueError(f"key '{key}' with data_type '{data_type}' not recognized. Check site IDs and data_type.")
    if info.empty:
      raise ValueError(f"key '{key}' not recognized. Valid keys are {list(special_keys.keys())}, site IDs, or None.")

  info = info[~(info['manual_error'].astype(str).str.startswith('x '))]
  all_sids = list(info['site_id'].unique())

  if key != 'test':
    # Remove test sids unless keyword test is passed
    all_sids = [sid for sid in all_sids if not sid.startswith('test')] 
    
  # Validate sids
  for sid in all_sids:
    if sid not in info['site_id'].values:
      raise ValueError(f"sid '{sid}' not found in info.csv")

  return all_sids

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

  for idx, row in info_df.iterrows():

    site = row['site_id']
    error = row['manual_error']

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

def read_info_df(extended=False, data_type=None, data_source=None, data_class=None, exclude_errors=False, error_type='manual_error', key=None, logger=None):
  import pandas
  from swerve import config
  CONFIG = config()
  file = CONFIG['files']['info_extended'] if extended else CONFIG['files']['info']
  print(f"    Reading {file}")
  info_df = pandas.read_csv(file)
  info_df['site_id'] = info_df['site_id'].astype(str)

  if key == 'paper_sids' or key == 'test_sids':
    if key not in CONFIG:
      raise ValueError(f"key '{key}' not found in config")
    key_sids = list(set(list(CONFIG[key]['GIC']['timeseries']) + list(CONFIG[key]['B']['timeseries'])))
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
    if error_type in info_df.columns:
      if logger is not None: 
        logger.info("    Excluding sites with {error_type}")
      if error_type == 'automated_error':
        info_df = info_df[~info_df['automated_error'].apply(lambda x: (isinstance(x, str) and x != '[]') or isinstance(x, float))]
      else:
        info_df = info_df[info_df[error_type].isna()]
    else:
      if logger is not None: 
        logger.info(f"    Error type {error_type} not available; Excluding sites with manual errors")
      info_df = info_df[info_df['manual_error'].isna()]

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

def format_cc_scatter(ax, regression=False):

  # Set the limits to be the same for both axes
  if not regression:
    max_x = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    max_y = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    max_xy = max(max_x, max_y)
    limits = [-max_xy, max_xy]
    # Sets the aspect ratio to make the plot square and ensure xlim and ylim are the same
    ax.set_aspect('equal', adjustable='box')
  else:
    limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]

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
  if not regression:
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

def fix_latex(df, data_type, formatters=None, index=False, note=None):
  # Defining column format
  if data_type =='B':
    column_format = "l " + " ".join(["p{1cm}"] * (len(df.columns) - 1))
  elif data_type == 'GIC':
    column_format = "l " + " ".join(["r"] * (len(df.columns) - 1))
  elif data_type == 'fit':
    column_format = "l l " + " ".join(["r"] * (len(df.columns) - 2))
  # Setting up string
  latex_string = df.to_latex(index=index, formatters=formatters, column_format=column_format)
  # Removing \toprule, \midrule, \bottomrule
  latex_string = latex_string.replace('\\toprule\n', '')
  latex_string = latex_string.replace('\\midrule\n','\\hline\n')
  if note:
    latex_string = latex_string.replace('\\bottomrule\n', f'{note}\n')
  else:
    latex_string = latex_string.replace('\\bottomrule\n', '')
  # Inserting \hline before the Mean row
  if 'Site ID' in df.keys() and 'Mean' in df['Site ID'].values:
    latex_string = latex_string.replace('Mean', '\\hline\nMean', 1)
  return latex_string

def savefig(base_dir, fname, logger, logger_indent=0, root_dir=None, fmts=['png','pdf']):
  import os
  logger_indent = " " * logger_indent

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