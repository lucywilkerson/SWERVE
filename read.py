import os
import sys
import pickle

from swerve import config, read_info_dict, read_site, plot_site

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

if len(sys.argv) > 1:
  sids_only = sys.argv[1:] # Only read these sites.
else:
  sids_only = None # Read all sites.
#sids_only = ['Union', 'Montgomery', 'Widows Creek', 'Bull Run']
#sids_only = ['10052', '10064']
#sids_only = ['10052', '10064']
#sids_only = ['Bull Run', '10052', '50100']
#sids_only = ['10233']

reparse = False    # Reparse the data files, even if they already exist.
show_plots = False # Show interactive plots as generated.
data_types = None  # Read and plot all data types.
data_types = ['B'] # Read and plot these data types only.

if sids_only is None:
  info = read_info_dict()
  sids_only = info.keys()

data = {}
errors = {}
for sid in sids_only:
  data[sid] = {}
  data[sid] = read_site(sid, data_types=data_types, logger=logger, reparse=reparse)
  plot_site(sid, data[sid], data_types=data_types, logger=logger, show_plots=show_plots)

if len(errors) > 0:
  logger.warning("Errors encountered while reading sites:")
  for sid, error in errors.items():
    logger.warning(f"  {sid}: {error}")

if sids_only is None:
  if not os.path.exists(os.path.dirname(CONFIG['files']['all'])):
    os.makedirs(os.path.dirname(FILES['all']))
  logger.info(f"Writing {CONFIG['files']['all']}")
  with open(CONFIG['files']['all'], 'wb') as f:
    pickle.dump(data, f)
