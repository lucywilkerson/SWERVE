# For debugging
reparse    = False  # Reparse the data files, even if they already exist (use if site_read.py modified).
data_types = 'B'   # Read and plot these data types. None => all data types.
add_errors = False # Add automated error checks to data and update info.extended files.

info_kwargs = {'data_type': data_types, # If specified, only return sites with this data type (e.g., GIC, B)
                 'data_source': None, # If specified, only return sites with this data source (e.g., TVA, NERC, SWMF)
                 'data_class': None, # If specified, only return sites with this data class (e.g., measured, calculated)
                 'exclude_errors': True # If True, excludes sites with known data issues (see info.csv 'manual_error' column)
              }

import utilrsw
from swerve import cli, config, sids, site_read, site_stats, site_stats_summary

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

# Crop the data :)
crop = True
start = '2024-05-10T12:00:00'
stop = '2024-05-11T12:00:00'

args = cli('main.py')
if args['sites'] is None:
  # Use command line arguments to specify sites under these conditions.
  sids_only = None # Read all sites.
else:
  sids_only = args['sites'].split(',')

# Get actual site IDs to process and validate given ones.
sids_only = sids(**info_kwargs, key=sids_only, logger=logger)

data = {}
stats = {}
rows = []
for sid in sids_only:
  data[sid] = {}

  # Read and parse data or use cached data if found and reparse is False.
  data[sid] = site_read(sid, data_types=data_types, start=start, stop=stop, logger=logger, reparse=reparse, add_errors=add_errors)

  # Add stats and metrics to data in data[sid] and returns what was added.
  stats[sid] = site_stats(sid, data[sid], data_types=data_types, logger=logger)

  utilrsw.print_dict(data[sid], indent=4)

if args['sites'] is None:
  if info_kwargs['exclude_errors']:
    # Create table of results
    site_stats_summary(stats, data_types=data_types, crop=crop, logger=logger)
