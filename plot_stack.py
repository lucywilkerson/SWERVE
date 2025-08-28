import os
import numpy
import pickle

from swerve import config, subset, read_info_dict, read_info_df
from swerve import plt_config, savefig, savefig_paper

import matplotlib.pyplot as plt

from datetick import datetick

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

DATA_DIR = CONFIG['dirs']['data']

base_dir = '_processed'

limits = CONFIG['limits']

plt_config()

def read(all_file, sid=None):
  info_dict = read_info_dict()
  info_df = read_info_df(exclude_errors=True, extended=True)

  print(f"Reading {all_file}")
  with open(all_file, 'rb') as f:
    data = pickle.load(f)

  return info_dict, info_df, data

def stack_plot_config(axes, data_with_offset, units, offset=40):
  plt.grid()
  plt.gca().yaxis.set_major_locator(plt.MultipleLocator(offset))
  #plt.legend(loc='upper right')
  plt.gca().yaxis.set_ticklabels([])  # Remove y-tick labels
  plt.gca().set_xlim(limits['plot'][0], limits['plot'][1])
  plt.gca().set_ylim(-offset, max(data_with_offset)+10)

  axes.spines['top'].set_visible(False)
  axes.spines['right'].set_visible(False)
  axes.spines['left'].set_visible(False)
  #axes.spines['bottom'].set_position(('outward', 10))  # Adjust position of x-axis
  axes.yaxis.set_ticks_position('none')  # Remove y-axis ticks
  datetick()

  # remove first x gridline
  xgridlines = axes.get_xgridlines()
  gridline_of_interest = xgridlines[0]
  gridline_of_interest.set_visible(False)

  # Add a vertical scale bar on the right side
  xmax = axes.get_xlim()[1]
  xbar = xmax - (xmax - axes.get_xlim()[0]) * 0.01  # 1% from right edge
  axes.plot([xbar, xbar], [0, -offset], color='k', linewidth=1, clip_on=False)
  # Add caps
  cap_width = 0.01 * (xmax - axes.get_xlim()[0])
  axes.plot([xbar - cap_width/2, xbar + cap_width/2], [0, 0], color='k', linewidth=1, clip_on=False)
  axes.plot([xbar - cap_width/2, xbar + cap_width/2], [-offset, -offset], color='k', linewidth=1, clip_on=False)
  # Add text label
  axes.text(xbar - cap_width, -offset/2, f'{offset} {units}', fontsize=plt.rcParams['ytick.labelsize'], verticalalignment='center', horizontalalignment='right')
  

def plot_all_gic(info, info_df, data_all, data_source=['TVA','NERC'], offset=40):
    # note NERC sites that are TVA duplicates
    units  = '[A]'
    sid_copies = CONFIG['sid_duplicates'] if 'sid_duplicates' in CONFIG else {}

    for source in data_source:
      logger.info(f"Plotting {source} GIC sites")
      sids = info_df[(info_df['data_source']==source)]['site_id'].unique()
      source_sites = {'sites': [], 'lat': [], 'lon': []}
      for sid in sids:
          sid_str = str(sid) # TODO: only necessary for October storm data
          if sid_str not in data_all.keys():
             logger.error(f"Site {sid} not found in data_all, rerun main.py to generate all.pkl")
          elif 'GIC' in data_all[sid_str] and source in info[sid_str]['GIC']['measured']:
              site_info = info_df[
                  (info_df['site_id'] == sid) & 
                  (info_df['data_type'] == 'GIC') & 
                  (info_df['data_class'] == 'measured')
              ]

              source_sites['sites'].append(sid_str)
              source_sites['lat'].append(site_info['mag_lat'].values)
              source_sites['lon'].append(site_info['mag_lon'].values)
      # Sort sites by latitude
      if not source_sites['lat'] and not source_sites['lon']:
          logger.warning(f"   No sites found for source {source}. Skipping.")
          continue
      sorted_sites = sorted(zip(source_sites['lat'], source_sites['sites'], source_sites['lon']))
      source_sites['lat'], source_sites['sites'], source_sites['lon'] = zip(*sorted_sites)

      if source == 'NERC':
        fig, axes = plt.subplots(1, 1, figsize=(8.5, 11))
      elif source == 'TVA':
        fig, axes = plt.subplots(1, 1, figsize=(8.5, 5))

      offset_fix = 0
      for i, sid in enumerate(source_sites['sites']):
          if sid in sid_copies.keys():
            offset_fix +=1
            continue # skipping NERC sites that are TVA duplicates
          if 'GIC' in data_all[sid].keys() and source in info[sid]['GIC']['measured']:
              time = data_all[sid]['GIC']['measured'][source]['original']['time']
              data = data_all[sid]['GIC']['measured'][source]['original']['data']

              # Subset to desired time range
              time, data = subset(time, data, limits['data'][0], limits['data'][1])

              # Add offset for each site
              data_with_offset = data + (i * offset) - (offset_fix * offset)

              # Plot the timeseries
              axes.plot(time, data_with_offset, linewidth=0.5)
              # Add text to the plot to label waveform
              sid_lat = source_sites['lat'][i].item()
              sid_lon = source_sites['lon'][i].item()
              text = f'{sid}\n({sid_lat:.1f},{sid_lon:.1f})'
              if sid in sid_copies.values():
                text = f'{sid}*\n({sid_lat:.1f},{sid_lon:.1f})'
              axes.text(limits['plot'][0], (i*offset)-(offset_fix*offset), text, fontsize=11, verticalalignment='center', horizontalalignment='left')
      stack_plot_config(axes, data_with_offset, units, offset=offset)
      # Save the figure
      fdir = os.path.join(base_dir, f'_{source.lower()}')
      savefig(fdir, f'gic_{source.lower()}', logger)
      savefig_paper(fdir, f'gic_{source.lower()}', logger) if 'paper' in CONFIG['dirs'] else None
      plt.close()


def plot_all_db(info, info_df, data_all, offset=400):
  units = '[nT]'
  info_df['site_id'] = info_df['site_id'].astype(str)
  info_df = info_df[(info_df['data_type']=='B')]
  sids = info_df[~(info_df['data_source']=='TEST')]['site_id'].unique()
  # note NERC sites that are TVA duplicates
  sid_copies = CONFIG['sid_duplicates'] if 'sid_duplicates' in CONFIG else {}

  logger.info("Plotting all dB sites")
  source_sites = {'sites': [], 'lat': [], 'lon': []}
  for sid in sids:
    if 'measured' in info[sid]['B']:
      site_info = info_df[
            (info_df['site_id'] == sid) & 
            (info_df['data_type'] == 'B') & 
            (info_df['data_class'] == 'measured')
        ]
      
      source_sites['sites'].append(sid)
      source_sites['lat'].append(site_info['mag_lat'].values[0])
      source_sites['lon'].append(site_info['mag_lon'].values[0])
  # Sort sites by latitude
  sorted_sites = sorted(zip(source_sites['lat'], source_sites['sites'], source_sites['lon']))
  source_sites['lat'], source_sites['sites'], source_sites['lon'] = zip(*sorted_sites)

  fig, axes = plt.subplots(1, 1, figsize=(8.5, 11))

  offset_fix = 0

  for i, sid in enumerate(source_sites['sites']):
    if sid in sid_copies.keys():
      offset_fix +=1
      continue # skipping NERC sites that are TVA duplicates
    if 'B' in data_all[sid].keys():
      source = info_df[
            (info_df['site_id'] == sid) & 
            (info_df['data_type'] == 'B') & 
            (info_df['data_class'] == 'measured')
        ]['data_source'].values[0] #TODO: find a simpler way to get data source
      time_b = data_all[sid]['B']['measured'][source]['modified']['time']
      data = data_all[sid]['B']['measured'][source]['modified']['data']

      # Subset to desired time range
      time_b, data = subset(time_b, data, limits['data'][0], limits['data'][1])
      data = numpy.linalg.norm(data, axis=1)

      # Add offset for each site
      data_with_offset = data + (i*offset) - (offset_fix*offset)

      # Plot the timeseries
      axes.plot(time_b, data_with_offset, linewidth=0.5)
      # Add text to the plot to label waveform
      sid_lat = source_sites['lat'][i]
      sid_lon = source_sites['lon'][i]
      text = f'{sid}\n({sid_lat:.1f},{sid_lon:.1f})'
      if sid in sid_copies.values():
        text = f'{sid}*\n({sid_lat:.1f},{sid_lon:.1f})'
      axes.text(limits['plot'][0], (i*offset)-(offset_fix*offset), text,
                fontsize=11, verticalalignment='center', horizontalalignment='left')
  stack_plot_config(axes, data_with_offset, units, offset=offset)
  # Save the figure
  fdir = os.path.join(base_dir, '_db')
  savefig(fdir, 'db_all', logger)
  savefig_paper(fdir, 'db_all', logger)
  plt.close()


info_dict, info_df, data_all = read(CONFIG['files']['all'])

plot_all_gic(info_dict, info_df, data_all)
plot_all_db(info_dict, info_df, data_all)
