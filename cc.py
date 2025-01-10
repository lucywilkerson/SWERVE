import os
import pickle

import numpy as np
import pandas as pd
import sys


data_dir = os.path.join('..', '2024-AGU-data')
sids = None # If none, plot all sites

fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)

# Filter out "bad" sites
filtered_df = info_df[~info_df['error'].str.contains('', na=False)]
filtered_df.reset_index(drop=True, inplace=True)
print(filtered_df)

sites = filtered_df['site_id'].tolist()
#sites = ['10052', '10238', '10255'] # For testing
print("site_1\tsite_2\tcc")

# This is still a work in progress
# I need to make user defined functions for some of it to make it easier to read
rows = []
for idx_1, site_1 in enumerate(sites):
  # Only comparing measured GIC for now
  if filtered_df['data_type'][idx_1] != 'GIC' or filtered_df['data_class'][idx_1] != 'measured':
    continue
  elif filtered_df['data_source'][idx_1] == 'NERC':
    #reading in data for site_1 if NERC
    fname = os.path.join(data_dir, 'processed', site_1, 'GIC_measured_NERC.pkl')
    with open(fname, 'rb') as f:
      #print(f"Reading {fname}")
      site_1_data = pickle.load(f)
    site_1_df = pd.DataFrame(site_1_data)
  else:
    #reading in data for site_1 if TVA
    site_1 = "".join(site_1.split()) #removing space from name to match file name
    fname = os.path.join(data_dir, 'processed', site_1, 'GIC_measured_TVA.pkl')
    with open(fname, 'rb') as f:
      #print(f"Reading {fname}")
      site_1_data = pickle.load(f)
    site_1_df = pd.DataFrame(site_1_data)
  for idx_2, site_2 in enumerate(sites):
    if idx_1 <= idx_2:  # Avoid duplicate pairs
      continue
    # Only comparing measured GIC for now
    elif filtered_df['data_type'][idx_2] != 'GIC' or filtered_df['data_class'][idx_2] != 'measured':
      continue
    elif filtered_df['data_source'][idx_2] == 'NERC':
      #reading in data for site_2 if NERC
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
    else:
      #reading in data for site_2 if TVA
      site_2 = "".join(site_2.split()) #removing space from name to match file name
      fname = os.path.join(data_dir, 'processed', site_2, 'GIC_measured_TVA.pkl')
      with open(fname, 'rb') as f:
        #print(f"Reading {fname}")
        site_2_data = pickle.load(f)
      site_2_df = pd.DataFrame(site_2_data)
      cc = 0.5  # for testing
      cc = np.corrcoef(site_1_df['original'][0]['data'], site_2_df['original'][0]['data'])[0, 1]
      #issue w cc calculation, need arrays to have same size 

      print(f"{site_1}\t{site_2}\t{cc}")
      rows.append([site_1, site_2, cc])


# Print the results again in order of decreasing correlation coefficient
rows = np.array(rows)
sorted_rows = rows[np.argsort(rows[:, 2])[::-1]]
print("site_1\tsite_2\tcc")
np.set_printoptions(threshold=sys.maxsize)
print(sorted_rows)


