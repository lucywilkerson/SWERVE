# For debugging
reparse    = False  # Reparse the data files, even if they already exist (use if site_read.py modified).
data_types = None   # Read and plot these data types. None => all data types.

info_kwargs = {'extended': False, # Should always be False, no need to use info.extended.csv
                 'data_type': data_types, # Specify in line 3, only return sites with this data type (e.g., GIC, B)
                 'data_source': None, # If specified, only return sites with this data source (e.g., TVA, NERC, SWMF)
                 'data_class': 'measured', # If specified, only return sites with this data class (e.g., measured, calculated)
                 'exclude_errors': True # If True, excludes sites with known data issues (see info.csv 'error' column)
              }

from swerve import cli, config, sids, site_read

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])


args = cli('main.py')
if args['sites'] is None:
  # Use command line arguments to specify sites under these conditions.
  sids_only = None # Read all sites.
else:
  sids_only = args['sites'].split(',')

sids_only = sids(**info_kwargs, key=sids_only)

GIC_data = {}
B_data = {}

for sid in sids_only:
  # Read and parse data or use cached data if found and reparse is False.
  data = site_read(sid, data_types=data_types, logger=logger, reparse=reparse)
  # Save GIC and B data separately
  for data_type in data.keys():
    if data_type == 'GIC':
      GIC_data[sid] = data
    if data_type == 'B':
      B_data[sid] = data

# Compute avg coherence between GIC site pairs and between B site pairs
def avg_coherence(data, data_type, fs=1/60, nperseg=256, show_plot=False):
    from scipy.signal import coherence
    import numpy as np

    if data_types is not None and data_type not in data_types:
       logger.warning(f"No data found for {data_type}.")
       return None

    # Prepare data for coherence calculation
    # TODO: some GIC sites still have nan values, not sure why
    def data_prep(sid1, sid2):
        prepped_data = [None, None]
        if 'measured' not in data[sid1][data_type] or 'measured' not in data[sid2][data_type]:
            return None, None
        for i, sid in enumerate([sid1, sid2]):
          for data_source in data[sid][data_type]['measured'].keys():
           if 'data' not in data[sid][data_type]['measured'][data_source]['modified']:
              prepped_data[i] = None
           else:
              prepped_data[i] = data[sid][data_type]['measured'][data_source]['modified']['data'].flatten()
        data1, data2 = prepped_data
        return data1, data2
    
    avg_C = []
    for i, sid in enumerate(data.keys()):
        for j, sid2 in enumerate(data.keys()):
            if j <= i:
                continue
            data1, data2 = data_prep(sid, sid2)
            if data1 is None or data2 is None:
              continue
            f, Cxy = coherence(data1, data2, fs=fs, nperseg=nperseg) # Coherence calculation
            avg_Cxy = np.mean(Cxy)
            avg_C.append(avg_Cxy)
            logger.info(f"Avg coherence between {sid} and {sid2} for {data_type}: {avg_Cxy}")

            # Plot coherence
            if show_plot:
                import matplotlib.pyplot as plt
                plt.semilogy(f, Cxy, 'k-')
                plt.title(f'Coherence between {sid} and {sid2} for {data_type}')
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Coherence')
                plt.grid()
                plt.show()
                plt.close()
    logger.info(f"{data_type} avg coherence range: {min(avg_C)} to {max(avg_C)}")
    return avg_C


GIC_coherence = avg_coherence(GIC_data, 'GIC')
B_coherence = avg_coherence(B_data, 'B')

