import os
import json
import numpy
import pickle
import datetime

import numpy as np
import pandas as pd

from swerve import config, subset, read_info_dict, read_info_df
from swerve import plt_config, savefig, savefig_paper, add_subplot_label

import matplotlib.pyplot as plt

from datetick import datetick

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

DATA_DIR = CONFIG['dirs']['data']

base_dir = '_processed'

limits = CONFIG['limits']['data']

def read(all_file, sid=None):
  info_dict = read_info_dict()
  info_df = read_info_df()

  print(f"Reading {all_file}")
  with open(all_file, 'rb') as f:
    data = pickle.load(f)

  return info_dict, info_df, data


def plot_all_gic(info, info_df, data_all, start, stop, data_source=['TVA', 'NERC'], offset=40):
    sids = info.keys()
    # note NERC sites that are TVA duplicates
    sid_copies = {'10197':'Sullivan',
                  '10204':'Shelby',
                  '10208':'Rutherford',
                  '10203':'Raccoon Mountain',
                  '10212':'Pinhook',
                  '10201':'Montgomery',
                  '10660':'Gleason',
                  '10200':'East Point',
                  '10207':'Bull Run'
                  }

    for source in data_source:
      print(f"Plotting {source} GIC sites")
      source_sites = {'sites': [], 'lat': [], 'lon': []}
      for sid in sids:
          if 'GIC' in data_all[sid] and source in info[sid]['GIC']['measured']:
              site_info = info_df[
                  (info_df['site_id'] == sid) & 
                  (info_df['data_type'] == 'GIC') & 
                  (info_df['data_class'] == 'measured')
              ]
              if site_info['error'].isna().all():
                  source_sites['sites'].append(sid)
                  source_sites['lat'].append(site_info['mag_lat'].values[0])
                  source_sites['lon'].append(site_info['mag_lon'].values[0])
      # Sort sites by latitude
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
              time = data_all[sid]['GIC']['measured'][0]['original']['time']
              data = data_all[sid]['GIC']['measured'][0]['original']['data']

              # Subset to desired time range
              time, data = subset(time, data, limits['data'][0], limits['data'][1])

              # Add offset for each site
              data_with_offset = data + (i * offset) - (offset_fix * offset)

              # Plot the timeseries
              axes.plot(time, data_with_offset, linewidth=0.5)
              # Add text to the plot to label waveform
              sid_lat = source_sites['lat'][i]
              sid_lon = source_sites['lon'][i]
              text = f'{sid}\n({sid_lat:.1f},{sid_lon:.1f})'
              if sid in sid_copies.values():
                text = f'{sid}*\n({sid_lat:.1f},{sid_lon:.1f})'
              axes.text(datetime.datetime(2024, 5, 10, 11, 0), (i*offset)-(offset_fix*offset), text, fontsize=9, verticalalignment='center', horizontalalignment='left')
      plt.grid()
      plt.gca().yaxis.set_major_locator(plt.MultipleLocator(offset))
      #plt.legend(loc='upper right')
      plt.gca().yaxis.set_ticklabels([])  # Remove y-tick labels
      plt.gca().set_xlim(limits['xlims'][0], limits['data'][1])
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

      # Save the figure
      fdir = os.path.join(base_dir, f'_{source.lower()}')
      savefig(fdir, f'gic_{source.lower()}', logger)
      savefig_paper(fdir, f'gic_{source.lower()}', logger)
      plt.close()


def plot_all_db(info, info_df, data_all, start, stop,  offset=400):
  sids = info.keys()
  # note NERC sites that are TVA duplicates
  sid_copies = {}

  print("Plotting all dB sites")
  source_sites = {'sites': [], 'lat': [], 'lon': []}
  for sid in sids:
    if 'B' in data_all[sid] and 'measured' in info[sid]['B']:
      site_info = info_df[
            (info_df['site_id'] == sid) & 
            (info_df['data_type'] == 'B') & 
            (info_df['data_class'] == 'measured')
        ]
      if site_info['error'].isna().all():
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
      time_b = data_all[sid]['B']['measured'][0]['modified']['time']
      data = data_all[sid]['B']['measured'][0]['modified']['data']

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
      axes.text(limits['xlims'][0], (i*offset)-(offset_fix*offset), text,
                fontsize=9, verticalalignment='center', horizontalalignment='left')

  plt.grid()
  plt.gca().yaxis.set_major_locator(plt.MultipleLocator(offset))
  #plt.legend(loc='upper right')
  plt.gca().yaxis.set_ticklabels([])  # Remove y-tick labels
  plt.gca().set_xlim(limits['xlims'][0], limits['data'][1])
  plt.gca().set_ylim(-offset, max(data_with_offset)+10)

  axes.spines['top'].set_visible(False)
  axes.spines['right'].set_visible(False)
  axes.spines['left'].set_visible(False)
  #axes.spines['bottom'].set_position(('outward', 10))  # Adjust position of x-axis
  axes.yaxis.set_ticks_position('none')  # Remove y-axis ticks
  datetick()

  # remove first x grid line
  xgridlines = axes.get_xgridlines()
  gridline_of_interest = xgridlines[0]
  gridline_of_interest.set_visible(False)

  # Save the figure
  fdir = os.path.join(base_dir, '_db')
  savefig(fdir, 'db_all', logger)
  savefig_paper(fdir, 'db_all', logger)
  plt.close()


info_dict, info_df, data_all = read(CONFIG['files']['all'])

plot_all_gic(info_dict, info_df, data_all, limits[0], limits[1])
plot_all_db(info_dict, info_df, data_all, limits[0], limits[1])
