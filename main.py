# Usage:
#   python main.py
#   python main.py paper
#   python main.py 'site1,site2,...'

sites  = None   # Read and plot data only sites in this array. None => all sites.
                    # Ignored if command line arguments are provided.

# For debugging
reparse    = True  # Reparse the data files, even if they already exist (use if site_read.py modified).
include_tests = True  # Include test sites in the read and plot.
show_plots = False  # Show interactive plots as generated.
data_types = None   # Read and plot these data types. None => all data types.
data_types = ['B']  # Read and plot these data types only.

import sys

import utilrsw
from swerve import config, sids, site_read, site_plot, site_stats

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

if sites is None and sys.argv is not None and len(sys.argv) > 1:
  # Use command line arguments to specify sites under these conditions.
  sids_only = sys.argv[1].split(',')
else:
  sids_only = None # Read all sites.

sids_only = sids(sids_only=sids_only)

if not include_tests:
  # Remove test sites from sids_only
  sids_only = [sid for sid in sids_only if not sid.startswith('test')]

# TODO: If info.extended.csv does not exist, run info.py code.
# data = read_info_dict() # Read info dictionary from info.extended.json file.

data = {}
stats = {}
rows = []
for sid in sids_only:
  data[sid] = {}

  # Read and parse data or use cached data if found and reparse is False.
  data[sid] = site_read(sid, data_types=data_types, logger=logger, reparse=reparse)

  # Add statistics to data in data[sid].
  stats[sid] = site_stats(sid, data[sid], data_types=data_types, logger=logger)

  utilrsw.print_dict(data[sid], indent=4)

  #site_plot(sid, data[sid], data_types=data_types, logger=logger, show_plots=show_plots)

def summary_stats(stats, logger=None):
  import pandas as pd

  for sid in stats.keys():
    for result in stats[sid].keys():
      if result.startswith('B') and 'calculated' in result and 'metrics' in stats[sid][result]:
        logger.info(f"  {sid}/{result} metrics: {stats[sid][result]['metrics']['pe'][0]:.3f}")
        rows.append({
          'site_id': sid,
          'model': result.split('/')[2],
          'pex': stats[sid][result]['metrics']['pe'][3],
          'ccx': stats[sid][result]['metrics']['cc'][3],
        })

  df = pd.DataFrame(rows)
  print(df)
  models = df['model'].unique()
  for model in models:
    model_df = df[df['model'] == model]
    mean_pex = model_df['pex'].mean()
    mean_ccx = model_df['ccx'].mean()
    mean_pex_se = model_df['pex'].std() / (len(model_df) ** 0.5)
    mean_ccx_se = model_df['ccx'].std() / (len(model_df) ** 0.5)
    logger.info(f"  Model: {model}, n = {len(model_df)}; Mean PE H: {mean_pex:.3f} +/- {mean_pex_se:0.3f}, Mean CC H: {mean_ccx:.3f} +/- {mean_ccx_se:0.3f}")

summary_stats(stats, logger=logger)

if sites is None and data_types is None and not include_tests:
  import utilrsw
  # Write data from all sites to a single file.
  utilrsw.write(CONFIG['files']['all'], data, logger=logger)
