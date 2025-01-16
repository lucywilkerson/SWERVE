import os
import sys
import pickle

import numpy as np
import pandas as pd
import numpy.ma as ma

from geopy.distance import geodesic


data_dir = os.path.join('..', '2024-AGU-data')

fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)

# Filter out sites with error message
# TODO: Also remove rows that don't have data_type = GIC and data_class = measured
# so that the keep() function is not needed.
info_df = info_df[~info_df['error'].str.contains('', na=False)]
info_df.reset_index(drop=True, inplace=True)

sites = info_df['site_id'].tolist()
#sites = ['10052', '10207'] # For testing

columns = ['site_1', 'site_2', 'cc', 'dist(km)']
print('\t'.join(columns))

#function to read TVA or NERC data to make below loop less complicated
def read_TVA_or_NERC(row):
  site_id = row['site_id']
  if row['data_source'] == 'NERC':
      #reading in data for site if NERC
      fname = os.path.join(data_dir, 'processed', site_id, 'GIC_measured_NERC.pkl')
  elif row['data_source'] == 'TVA':
      #reading in data for site if TVA
      site_id = "".join(site_id.split()) #removing space from name to match file name
      fname = os.path.join(data_dir, 'processed', site_id, 'GIC_measured_TVA.pkl')

  with open(fname, 'rb') as f:
      #print(f"Reading {fname}")
      site_data = pickle.load(f)

  site_df = pd.DataFrame(site_data)
  mod_data = site_df['modified'][0]['data']
  masked_data = ma.masked_invalid(mod_data)
  return mod_data, masked_data

def site_distance(df, idx_1, idx_2):
  dist = geodesic((df['geo_lat'][idx_1], df['geo_lon'][idx_1]), 
                  (df['geo_lat'][idx_2], df['geo_lon'][idx_2])).km
  return dist

def keep(row):
  # Only compare GIC/measured
  return row['data_type'] == 'GIC' and row['data_class'] == 'measured'

rows = []
for idx_1, row in info_df.iterrows():

  site_1_id = row['site_id']
  if not keep(row) or site_1_id not in sites:
    continue

  site_1_data, msk_site_1_data = read_TVA_or_NERC(row)

  for idx_2, row in info_df.iterrows():
    if idx_1 <= idx_2:  # Avoid duplicate pairs
      continue

    site_2_id = row['site_id']

    if not keep(row) or site_2_id not in sites:
      continue

    site_2_data, msk_site_2_data = read_TVA_or_NERC(row)

    cov = ma.corrcoef(msk_site_1_data, msk_site_2_data)
    cc = cov[0, 1]
    if np.isnan(cc):
      continue

    # Compute distance between sites in km
    distance = site_distance(info_df, idx_1, idx_2)

    print(f"{site_1_id}\t{site_2_id}\t{cc:+.2f}\t{distance:6.1f}")
    rows.append([site_1_id, site_2_id, cc, distance])


# Print the results again in order of decreasing correlation coefficient
rows_df = pd.DataFrame(rows, columns=columns)
sorted_rows = rows_df.sort_values(by='cc', ascending=False)
#np.set_printoptions(threshold=sys.maxsize)

output_fname = os.path.join(data_dir, '_results', 'cc.pkl')
if not os.path.exists(os.path.dirname(output_fname)):
  os.makedirs(os.path.dirname(output_fname))
print(f"Saving {output_fname}")
with open(output_fname, 'wb') as f:
  pickle.dump(sorted_rows, f)
print(f"Saved {output_fname}")
