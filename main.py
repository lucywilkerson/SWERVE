# Usage:
#   python main.py
#   python main.py paper
#   python main.py 'site1,site2,...'

import sys

from swerve import config, sids, site_read, site_plot, site_stats

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
reparse    = True   # Reparse the data files, even if they already exist.
sids_only  = None   # Read and plot data only sites in this array. None => all sites.
show_plots = False  # Show interactive plots as generated.
data_types = None   # Read and plot all data types. None means => data types.
data_types = ['B']  # Read and plot these data types only.

if sids_only is None and sys.argv is not None and len(sys.argv) > 1:
  # Use command line arguments to specify sites under these conditions.
  sids_only = sys.argv[1].split(',')
else:
  sids_only = None # Read all sites.

sids_only = sids(sids_only=sids_only)

data = {}
for sid in sids_only:
  data[sid] = {}

  # Read and parse data or use cached data if found and reparse is False.
  data[sid] = site_read(sid, data_types=data_types, logger=logger, reparse=reparse)

  # Add statistics to data in data[sid].
  #site_stats(sid, data[sid], data_types=data_types, logger=logger)

  #site_plot(sid, data[sid], data_types=data_types, logger=logger, show_plots=show_plots)


if sids_only is None and data_types is None:
  import utilrsw
  # Write data from all sites to a single file.
  utilrsw.write(CONFIG['files']['all'], data, logger=logger)
