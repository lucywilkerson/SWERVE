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

base_dir = 'data_processed/summary'

limits = CONFIG['limits']

plt_config()

def read(all_file, sid=None):
  info_dict = read_info_dict()
  info_df = read_info_df(exclude_errors=True, extended=True)

  print(f"Reading {all_file}")
  with open(all_file, 'rb') as f:
    data = pickle.load(f)

  return info_dict, info_df, data

def stack_plot_config(axes, data_with_offset, sites_plotted, units, offset=40):
  plt.grid()
  plt.gca().yaxis.set_major_locator(plt.MultipleLocator(offset))
  #plt.legend(loc='upper right')
  plt.gca().yaxis.set_ticklabels([])  # Remove y-tick labels
  plt.gca().set_xlim(limits['plot'][0], limits['plot'][1])
  if numpy.all(numpy.isnan(data_with_offset)):
     data_with_offset = [sites_plotted*offset]  # to avoid error below
  plt.gca().set_ylim(-offset, numpy.nanmax(data_with_offset)+10)

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


def plot_intermag(offset=1000, use_hapi=False):
  units = '(nT)'
  import pandas as pd
  import csv as _csv
  csv_file = os.path.join('..', 'C-SWIM','data','geomag_data','combined_geomag_df.csv')
  if csv_file:
        logger.info(f"Reading INTERMAG CSV: {csv_file}")
        if pd is not None:
            intermag_df = pd.read_csv(csv_file)
        else:
            # Fallback to built-in csv module if pandas is not available
            with open(csv_file, newline='') as _f:
                reader = _csv.DictReader(_f)
                intermag_df = list(reader)
  sids = intermag_df['site_id'].unique()
    

  logger.info("Plotting all INTERMAG sites")
  source_sites = {'sites': [], 'lat': [], 'lon': []}
  for sid in sids:
    site_df = intermag_df[intermag_df['site_id'] == sid]
      
    source_sites['sites'].append(sid)
    source_sites['lat'].append(site_df['Latitude'].values[0])
    source_sites['lon'].append(site_df['Longitude'].values[0])
  # Sort sites by latitude
  sorted_sites = sorted(zip(source_sites['lat'], source_sites['sites'], source_sites['lon']))
  source_sites['lat'], source_sites['sites'], source_sites['lon'] = zip(*sorted_sites)

  if use_hapi:
    logger.info("Using HAPI INTERMAG data")
    from hapiclient import hapi, hapitime2datetime
    import pandas
    # Read INTERMAG data from HAPI server
    server     = 'https://imag-data.bgs.ac.uk/GIN_V1/hapi'
    intermag_df = pandas.DataFrame()
    for sid in sids:
      sid_lower = sid.lower()
      dataset    = f'{sid_lower}/best-avail/PT1M/xyzf'
      parameters = 'Field_Vector'
      start      = '2024-05-10T00:01:00Z' # min 1992-01-01T00:01:00Z
      stop       = '2024-05-13T00:01:00.000Z' # max 2011-12-31T23:59:00Z

      data, meta = hapi(server, dataset, parameters, start, stop)

      dfs = []
      dfs.append(pandas.DataFrame({'Timestamp': hapitime2datetime(data['Time'])}))
      
      field_vector = data['Field_Vector']
      for i, component in enumerate(['X', 'Y', 'Z']):
        dfs.append(pandas.DataFrame({component: field_vector[:, i]}))
        fill = meta['parameters'][1]['fill']
        if fill is not None:
          fill = float(fill)
          dfs[-1][dfs[-1][component] == fill] = numpy.nan  # remove erroneous high values
        dfs[-1][component] = dfs[-1][component] - numpy.median(dfs[-1][component])  # remove median offset
      
      df = pandas.concat(dfs, axis=1)

      intermag_df = pandas.concat([intermag_df, df.assign(site_id=sid)], ignore_index=True)

  for orientation in ['X', 'Y', 'Z']:
    fig, axes = plt.subplots(1, 1, figsize=(8.5, 11))
    for i, sid in enumerate(source_sites['sites']):
      site_df = intermag_df[intermag_df['site_id'] == sid]
      time_b = site_df['Timestamp'].values
      data = site_df[orientation].values.astype(float)

      # Subset to desired time range
      time_b, data = subset(time_b, data, limits['data'][0], limits['data'][1])

      # Add offset for each site
      data_with_offset = data + (i*offset)

      # Plot the timeseries
      axes.plot(time_b, data_with_offset, linewidth=0.5)
      # Add text to the plot to label waveform
      sid_lat = source_sites['lat'][i]
      sid_lon = source_sites['lon'][i]
      text = f'{sid}\n({sid_lat:.1f},{sid_lon:.1f})'
      axes.text(limits['plot'][0], (i*offset), text,
                fontsize=11, verticalalignment='center', horizontalalignment='left')
    sites_plotted=len(source_sites['sites'])
    stack_plot_config(axes, data_with_offset, sites_plotted, units, offset=offset)
    # Save the figure
    fdir = os.path.join(base_dir, '_db')
    if use_hapi:
      savefig(fdir, f'intermag_hapi_{orientation}', logger)
    else:
      savefig(fdir, f'intermag_{orientation}', logger)
    plt.close()


plot_intermag(use_hapi=True)