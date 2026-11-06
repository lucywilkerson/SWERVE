"""Plot data availability for each data source.

For each data source, a horizontal bar is plotted for every time window with available data.
INTERMAG is shown as a continuous bar spanning the full date range.

Availability windows are determined by reading the 'original' time from cached
pkl files in data_processed. If no processed cache exists for a event-partition source, 
the availability window just uses the event date(s).

Availability windows are cached in data_processed/summary/source_availability.json.
reparse = True will rewrite source_availability.json.

Usage:
  python swerve/plot/plot_sources.py                      # all catalog sources
  python swerve/plot/plot_sources.py --config nerc.yaml   # sources from run config
"""

import os
import glob
import json
import pickle
import datetime

import numpy
import pandas
from matplotlib import pyplot as plt

from swerve import config, plt_config
from swerve.util.source_catalog import ANNUAL_SOURCES, CONTINUOUS_SOURCES, EVENT_PARTITIONED_SOURCES, source_kind, source_partitions

reparse = False # If True, rescan processed data files even if availability cache exists.

EVENT_FALLBACK_DAYS = 3 # Window length used when no processed cache exists for an event partition.

PLOT_START = datetime.datetime(2003, 1, 1)
PLOT_STOP = datetime.datetime(2027, 1, 1)

CONFIG = config()

paper_dir = os.path.join(CONFIG['dirs']['data'], '..', 'SWERVE_data_paper')

if 'logger' in CONFIG:
  logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
  log = logger.info
  log_warning = logger.warning
else:
  # config() returns a dirs-only dict when no --config is given.
  log = print
  log_warning = print

plt_config()

main_events = {
    'gannon_storm': {
        'date': '2024-05-10',
        'label': 'May 2024 \n"Gannon" storm',
        'color': 'm'
    },
    'st_pat_storm': {
        'date': '2015-03-17',
        'label': 'March 2015 \n"St. Patrick" storm',
        'color': 'g'
    }
}


def _select_sources():
  """All catalog sources by default; restrict to run config data_source if given."""
  data_sources = None
  if 'info_kwargs' in CONFIG:
    data_sources = CONFIG['info_kwargs']['data_source']
  if data_sources is None:
    data_sources = EVENT_PARTITIONED_SOURCES | ANNUAL_SOURCES | {'INTERMAG'}
  elif isinstance(data_sources, str):
    data_sources = [data_sources]
  return sorted(set(data_sources) - (CONTINUOUS_SOURCES - {'INTERMAG'}))


def _original_time_range(pkl_file, data_source):
  """Return (start, stop) of the 'original' time array in a processed pickle."""
  try:
    with open(pkl_file, 'rb') as f:
      cached = pickle.load(f)
  except Exception as error:
    log_warning(f"    Could not read '{pkl_file}': {error}")
    return None
  # Event pickles are {data_source: {'original': {...}, ...}}
  # Non-event pickles are the original data dict itself
  if isinstance(cached, dict) and 'time' in cached:
    orig = cached
  else:
    entry = cached.get(data_source) if isinstance(cached, dict) else None
    orig = entry.get('original') if isinstance(entry, dict) else None
  if not isinstance(orig, dict) or 'error' in orig or 'time' not in orig:
    return None
  try:
    times = numpy.asarray(orig['time'], dtype='datetime64[ns]')
    times = times[~numpy.isnat(times)]
  except (TypeError, ValueError):
    times = pandas.to_datetime(pandas.Series(orig['time']), errors='coerce').dropna().values
  if times.size == 0:
    return None
  return (pandas.Timestamp(times.min()).to_pydatetime(),
          pandas.Timestamp(times.max()).to_pydatetime())


def _scan_partition(processed_dir, data_source, kind, partition):
  """Scan processed caches of all sites for one partition; return (start, stop) or None."""
  if kind == 'event':
    pattern = os.path.join(processed_dir, partition, 'sites', '*', 'data', f'*_{data_source}.pkl')
  else:
    pattern = os.path.join(processed_dir, 'non-event', partition, '*', f'*_{data_source}.pkl')

  start, stop = None, None
  for pkl_file in sorted(glob.glob(pattern)):
    time_range = _original_time_range(pkl_file, data_source)
    if time_range is None:
      continue
    if start is None or time_range[0] < start:
      start = time_range[0]
    if stop is None or time_range[1] > stop:
      stop = time_range[1]

  if start is None:
    return None
  return start, stop


def _fallback_window(kind, partition):
  """Window from the partition name when no processed cache exists."""
  if kind == 'event':
    start = datetime.datetime.strptime(partition, '%Y-%m-%d')
    return start, start + datetime.timedelta(days=EVENT_FALLBACK_DAYS)
  year = int(partition)
  return datetime.datetime(year, 1, 1), datetime.datetime(year + 1, 1, 1) - datetime.timedelta(seconds=1)


def availability(data_sources):
  """Dict of {source: {partition: {'start', 'stop', 'method'}}}.

  'method' is 'scanned' (from processed data files) or 'fallback' (from
  partition names), or 'continuous' (INTERMAG across the plotted date range).
  """
  original_dir = CONFIG['dirs']['original']
  processed_dir = CONFIG['dirs']['processed']

  cache_file = os.path.join(processed_dir, 'summary', 'source_availability.json')
  cache = {}
  if not reparse and os.path.exists(cache_file):
    log(f"Reading availability cache {cache_file}")
    with open(cache_file, 'r') as f:
      cache = json.load(f)

  windows = {}
  for data_source in data_sources:
    if data_source == 'INTERMAG':
      windows[data_source] = {
        'continuous': {
          'start': PLOT_START.isoformat(),
          'stop': PLOT_STOP.isoformat(),
          'method': 'continuous',
        }
      }
      continue
    kind = source_kind(data_source)
    try:
      partitions = source_partitions(original_dir, data_source)
    except FileNotFoundError as error:
      log_warning(f"  {error} Skipping.")
      continue

    log(f"{data_source} ({kind}): {len(partitions)} partition(s)")
    windows[data_source] = {}
    for partition in partitions:
      cached = cache.get(data_source, {}).get(partition)
      if cached is not None:
        windows[data_source][partition] = cached
        continue

      time_range = _scan_partition(processed_dir, data_source, kind, partition)
      if time_range is None:
        log_warning(f"  {partition}: no processed cache found; using partition dates.")
        time_range = _fallback_window(kind, partition)
        method = 'fallback'
      else:
        log(f"  {partition}: {time_range[0]} to {time_range[1]}")
        method = 'scanned'

      windows[data_source][partition] = {
        'start': time_range[0].isoformat(),
        'stop': time_range[1].isoformat(),
        'method': method,
      }

    if not windows[data_source]:
      log_warning(f"  No partitions found for '{data_source}'. Skipping.")
      del windows[data_source]

    # Write cache incrementally so a long scan is not lost if interrupted.
    cache.update(windows)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as f:
      json.dump(cache, f, indent=2)

  log(f"Wrote availability cache {cache_file}")
  return windows


def plot_availability(windows, add_events=None):
  import matplotlib.dates as mdates
  sources = sorted(windows, key=str.lower)

  all_times = [
    datetime.datetime.fromisoformat(window[key])
    for source in sources
    for window in windows[source].values()
    for key in ('start', 'stop')
  ]
  # Pad short windows to a min drawn width so they remain visible
  min_width = (max(all_times) - min(all_times)) * 0.004

  fig, ax = plt.subplots(figsize=(10, 0.4*len(sources) + 1.5))

  if add_events is not None:
    for event in add_events:
      ax.axvline(
          x=datetime.datetime.fromisoformat(event['date']), 
          color=event.get('color', 'm'), 
          linestyle='--', 
          linewidth=1
      )
      ax.text(
          datetime.datetime.fromisoformat(event['date'])+datetime.timedelta(days=30), 
          -0.3, 
          event['label'], 
          color=event.get('color', 'm'), 
          rotation=0, 
          verticalalignment='top'
      )

  methods_used = set()
  for y, source in enumerate(sources):
    for window in windows[source].values():
      start = datetime.datetime.fromisoformat(window['start'])
      stop = datetime.datetime.fromisoformat(window['stop'])
      if stop - start < min_width:
        center = start + (stop - start)/2
        start, stop = center - min_width/2, center + min_width/2
      methods_used.add(window['method'])
      ax.hlines(y, start, stop, color='k', linewidth=6)
  
  ax.set_yticks(range(len(sources)))
  ax.set_yticklabels(sources)
  ax.set_ylim(len(sources) - 0.5, -0.5) # First source alphabetically at top
  ax.grid()
  ax.set_title('Data availability by source')

  ax.xaxis.set_major_locator(mdates.AutoDateLocator())
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
  ax.xaxis.set_minor_locator(mdates.YearLocator())

  ax.set_xlim(PLOT_START, PLOT_STOP)
  plt.savefig(os.path.join(paper_dir, 'availability_plot.png'), bbox_inches='tight')
  plt.savefig(os.path.join(paper_dir, 'availability_plot.pdf'), bbox_inches='tight')


if __name__ == '__main__':
  data_sources = _select_sources()
  log(f"Data sources: {', '.join(data_sources)}")
  windows = availability(data_sources)
  plot_availability(windows, add_events=main_events.values())
