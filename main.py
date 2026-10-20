# Usage:
#   python main.py -h
#   python main.py --sites paper
#   python main.py --config test.yaml --sites test
#   python main.py --sites 'site1,site2,...'

# For debugging
reparse    = True  # Reparse the data files, even if they already exist (use if site_read.py modified).
show_plots = False  # Show interactive plots as generated.
add_errors = True # Add automated error checks to data and update info.extended files.
summary_plots = False # Create and save plots for full run (e.g. map, stack plots, OMNI data)

import os
import utilrsw
from swerve import cli, config, sids, site_read, site_plot, site_stats, site_stats_summary
#from swerve.plot import plot_maps, plot_stack

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
info_kwargs = CONFIG['info_kwargs']
data_types = info_kwargs['data_type']
events = CONFIG['event']

# Ensure info file exists
info_file = CONFIG['files']['info']
if not os.path.exists(info_file):
  import runpy
  runpy.run_path(os.path.join(os.path.dirname(__file__), 'info.py'), run_name='__main__')

args = cli('main.py')
if args['sites'] is None:
  # Use command line arguments to specify sites under these conditions.
  sids_only = None # Read all sites.
else:
  sids_only = args['sites'].split(',')

# Get actual site IDs to process and validate given ones.
sids_only = sids(**info_kwargs, key=sids_only, add_event=True, logger=logger)

# data = read_info_dict() # Read info dictionary from info.extended.json file.

data = {}
stats = {}
rows = []
for sid, event in sids_only:
  data.setdefault(event, {})
  stats.setdefault(event, {})

  # Read and parse data or use cached data if found and reparse is False.
  data[event][sid] = site_read(sid, event, data_types=data_types, logger=logger, reparse=reparse, add_errors=add_errors)

  if CONFIG['event'][event].get('event_type', 'event') == 'annual':
    logger.info(f"Skipping statistics and plots for annual data from '{sid}', {event}.")
    continue

  # Add stats and metrics to data[event][sid] and return what was added.
  stats[event][sid] = site_stats(sid, data[event][sid], data_types=data_types, logger=logger)

  utilrsw.print_dict(data[event][sid], indent=4)

  site_plot(sid, event, data[event][sid], data_types=data_types, logger=logger, show_plots=show_plots)

if args['sites'] is None:
  import utilrsw
  if add_errors and info_kwargs['exclude_errors']==None:
    # Update info.extended.csv and info.extended.json with automated errors
    from swerve import update_info_extended
    update_info_extended(sids_only, data, logger=logger, CONFIG=CONFIG)
  if info_kwargs['exclude_errors'] == None and data_types is None:
    # Write data from all sites to a single file.
    utilrsw.write(CONFIG['files']['all'], data, logger=logger)

if CONFIG['main_kwargs']['summary_table'] and len(events) == 1:
  event = next(iter(events))
  if CONFIG['event'][event].get('event_type', 'event') == 'annual':
    logger.info("Skipping summary table because the configured event is annual data.")
  else:
    # Create table of results
    site_stats_summary(stats[event], data_types=data_types, logger=logger)
elif CONFIG['main_kwargs']['summary_table']:
  logger.info("Skipping summary table because multiple events were requested.")

#if summary_plots:
#  from swerve.plot import plot_sw_and_indices

