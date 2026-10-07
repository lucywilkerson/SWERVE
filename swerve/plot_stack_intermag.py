import os

from swerve import config, subset
from swerve import plt_config, savefig
from swerve.plot_stack import read, stack_plot_config
from swerve.read.read_intermag import read_intermag

import matplotlib.pyplot as plt

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

base_dir = 'data_processed/summary'

limits = CONFIG['limits']

plt_config()


def plot_intermag(intermag_df, source_sites, offset=1000):
  units = '(nT)'

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
    stack_plot_config(axes, data_with_offset, units, offset=offset, sites_plotted=sites_plotted)
    # Save the figure
    fdir = os.path.join(base_dir, '_db')
    savefig(fdir, f'intermag_hapi_{orientation}', logger)
    plt.close()