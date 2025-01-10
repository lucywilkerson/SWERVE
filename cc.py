import os
import pickle

import numpy as np
import pandas as pd


data_dir = os.path.join('..', '2024-AGU-data')
sids = None # If none, plot all sites

fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)

# Filter out "bad" sites
filtered_df = info_df[~info_df['error'].str.contains('', na=False)]

print(filtered_df)

sites = ['10052', '10238', '10255'] # For testing
print("site_1\tsite_2\tcc")

rows = []
for idx_1, site_1 in enumerate(sites):
  #reading in data for site_1
  fname = os.path.join(data_dir, 'processed', site_1, 'GIC_measured_NERC.pkl')
  with open(fname, 'rb') as f:
    #print(f"Reading {fname}")
    site_1_data = pickle.load(f)
  site_1_df = pd.DataFrame(site_1_data)
  for idx_2, site_2 in enumerate(sites):
    if idx_1 <= idx_2:  # Avoid duplicate pairs
      continue
    else:
      fname = os.path.join(data_dir, 'processed', site_2, 'GIC_measured_NERC.pkl')
      with open(fname, 'rb') as f:
        #print(f"Reading {fname}")
        site_2_data = pickle.load(f)
      site_2_df = pd.DataFrame(site_2_data)
      cc = 1.0  # for testing
      cc = np.corrcoef(site_1_df['original'][0]['data'], site_2_df['original'][0]['data'])[0, 1]
      #issue w cc calculation, need arrays to have same size 
      print(f"{site_1}\t{site_2}\t{cc}")
      rows.append([site_1, site_2, cc])


# TODO: Print the results again in order of decreasing correlation coefficient


