# Usage:
#   python main.py
#   python main.py paper
#   python main.py test
#   python main.py 'site1,site2,...'

sites  = None   # Read and plot data only sites in this array. None => all sites.
                    # Ignored if command line arguments are provided.

# For debugging
reparse    = False  # Reparse the data files, even if they already exist (use if site_read.py modified).
show_plots = False  # Show interactive plots as generated.
data_types = None   # Read and plot these data types. None => all data types.
data_types = ['B']  # Read and plot these data types only.

import sys

import utilrsw
from swerve import config, sids, site_read, site_plot, site_stats, site_stats_summary

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

if sites is None and sys.argv is not None and len(sys.argv) > 1:
  # Use command line arguments to specify sites under these conditions.
  sids_only = sys.argv[1].split(',')
else:
  sids_only = None # Read all sites.

sids_only = sids(sids_only=sids_only)

# TODO: If info.extended.csv does not exist, run info.py code.
# data = read_info_dict() # Read info dictionary from info.extended.json file.

data = {}
stats = {}
rows = []
for sid in sids_only:
  data[sid] = {}

  # Read and parse data or use cached data if found and reparse is False.
  data[sid] = site_read(sid, data_types=data_types, logger=logger, reparse=reparse)

  # Add stats and metrics to data in data[sid] and returns what was added.
  stats[sid] = site_stats(sid, data[sid], data_types=data_types, logger=logger)

  utilrsw.print_dict(data[sid], indent=4)

  site_plot(sid, data[sid], data_types=data_types, logger=logger, show_plots=show_plots)

#site_stats_summary(stats, logger=logger)

if sites is None and data_types is None:
  import utilrsw
  # Write data from all sites to a single file.
  utilrsw.write(CONFIG['files']['all'], data, logger=logger)
