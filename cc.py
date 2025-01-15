import os
import pickle

import numpy as np
import pandas as pd
import sys
from geopy.distance import geodesic


data_dir = os.path.join('..', '2024-AGU-data')
sids = None # If none, plot all sites

fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)


# Filter out "bad" sites
filtered_df = info_df[~info_df['error'].str.contains('', na=False)]
filtered_df.reset_index(drop=True, inplace=True)

sites = filtered_df['site_id'].tolist()
#sites = ['10052', '10238', '10255'] # For testing
print("site_1\tsite_2\tcc\tdist(km)")

#function to read TVA or NERC data to make below loop less complicated
def read_TVA_or_NERC (df,site_id):
  if df['data_source'][idx_1] == 'NERC':
      #reading in data for site if NERC
      fname = os.path.join(data_dir, 'processed', site_id, 'GIC_measured_NERC.pkl')
  elif df['data_source'][idx_1] == 'TVA':
      #reading in data for site if TVA
      site_id = "".join(site_id.split()) #removing space from name to match file name
      fname = os.path.join(data_dir, 'processed', site_id, 'GIC_measured_TVA.pkl')
  with open(fname, 'rb') as f:
      #print(f"Reading {fname}")
      site_data = pickle.load(f)
  site_df = pd.DataFrame(site_data)
  return site_df


rows = []
for idx_1, site_1 in enumerate(sites):
  # Only comparing measured GIC for now
  if filtered_df['data_type'][idx_1] != 'GIC' or filtered_df['data_class'][idx_1] != 'measured':
    continue
  else:
    site_1_df = read_TVA_or_NERC(filtered_df,site_1)
  for idx_2, site_2 in enumerate(sites):
    if idx_1 <= idx_2:  # Avoid duplicate pairs
      continue
    # Only comparing measured GIC for now
    elif filtered_df['data_type'][idx_2] != 'GIC' or filtered_df['data_class'][idx_2] != 'measured':
      continue
    else: 
      site_2_df = read_TVA_or_NERC(filtered_df,site_2)
      cc = np.corrcoef(site_1_df['modified'][0]['data'], site_2_df['modified'][0]['data'])[0, 1]
      #issue w cc calculation, get rid of nan values
      #finding distange between sites in km
      dist = geodesic((filtered_df['geo_lat'][idx_1], filtered_df['geo_lon'][idx_1]), 
                      (filtered_df['geo_lat'][idx_2], filtered_df['geo_lon'][idx_2])).km
      print(f"{site_1}\t{site_2}\t{cc}\t{dist}")
      rows.append([site_1, site_2, cc, dist])



# Print the results again in order of decreasing correlation coefficient
rows_df = pd.DataFrame(rows, columns=['site_1', 'site_2', 'cc', 'dist(km)'])
sorted_rows = rows_df.sort_values(by='cc', ascending=False)
np.set_printoptions(threshold=sys.maxsize)
print(sorted_rows)


