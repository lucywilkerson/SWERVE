# Usage:
#   python main.py
#   python main.py paper
#   python main.py test
#   python main.py 'site1,site2,...'

# For debugging
reparse    = False  # Reparse the data files, even if they already exist (use if site_read.py modified).
show_plots = False  # Show interactive plots as generated.
data_types = None   # Read and plot these data types. None => all data types.

import sys

import utilrsw
from swerve import cli, config, sids, site_read, site_plot, site_stats, site_stats_summary

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

args = cli('main.py')
if args['sites'] is None:
  # Use command line arguments to specify sites under these conditions.
  sids_only = None # Read all sites.
else:
  sids_only = sys.argv[1].split(',')

# Get actual site IDs to process and validate given ones.
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

  #site_plot(sid, data[sid], data_types=data_types, logger=logger, show_plots=show_plots)

#dfs = site_stats_summary(stats, data_types=data_types, logger=logger)

if args['sites'] is None and data_types is None:
  import utilrsw
  # Write data from all sites to a single file.
  utilrsw.write(CONFIG['files']['all'], data, logger=logger)
