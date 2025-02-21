import os
import sys
import pickle

import numpy as np
import pandas as pd
import numpy.ma as ma

from geopy.distance import geodesic

data_dir = os.path.join('..', '2024-AGU-data', '_processed')
out_dir = os.path.join('..', '2024-AGU-data', '_results')

fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)

# Filter out sites with error message
# Also remove rows that don't have data_type = GIC and data_class = measured
info_df = info_df[~info_df['error'].str.contains('', na=False)]
info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
info_df.reset_index(drop=True, inplace=True)
#print(info_df)

sites = info_df['site_id'].tolist()
sites = ['10052', '10207'] # For testing
sites = ['Widows Creek', 'Bull Run'] # For testing

columns = ['site_1', 'site_2', 'cc', 'dist(km)', 'bad_1', 'bad_2', 'std_1', 'std_2', 'beta_diff']
print('\t'.join(columns))

def write_table(rows, out_dir):
  # Print the results again in order of decreasing correlation coefficient
  df = pd.DataFrame(rows, columns=columns)
  df = df.sort_values(by='cc', ascending=False)

  output_fname = os.path.join(out_dir, 'cc.pkl')
  if not os.path.exists(os.path.dirname(output_fname)):
    os.makedirs(os.path.dirname(output_fname))

  print(f"Writing {output_fname}")
  with open(output_fname, 'wb') as f:
    pickle.dump(df, f)

  # Write the DataFrame to a markdown file
  output_fname = os.path.join(out_dir, 'cc.md')
  print(f"Writing {output_fname}")
  with open(output_fname, 'w') as f:
    f.write(df.to_markdown(index=False))


def read_TVA_or_NERC(row):
  site_id = row['site_id']
  if row['data_source'] == 'NERC':
      #reading in data for site if NERC
      fname = os.path.join(data_dir, site_id, 'GIC_measured_NERC.pkl')
  elif row['data_source'] == 'TVA':
      #reading in data for site if TVA
      site_id = "".join(site_id.split()) #removing space from name to match file name
      fname = os.path.join(data_dir, site_id, 'GIC_measured_TVA.pkl')

  with open(fname, 'rb') as f:
      #print(f"Reading {fname}")
      site_data = pickle.load(f)

  site_df = pd.DataFrame(site_data)
  mod_data = site_df['modified'][0]['data'] # 1-min avg data
  masked_data = ma.masked_invalid(mod_data) # 1-min data w nan values masked
  return mod_data, masked_data


def site_distance(df, idx_1, idx_2):
  dist = geodesic((df['geo_lat'][idx_1], df['geo_lon'][idx_1]), 
                  (df['geo_lat'][idx_2], df['geo_lon'][idx_2])).km
  return dist


rows = []
for idx_1, row in info_df.iterrows():

  site_1_id = row['site_id']
  if site_1_id not in sites:
    continue

  site_1_data, msk_site_1_data = read_TVA_or_NERC(row)

  # finding number of nans masked
  bad_1 = np.sum(msk_site_1_data.mask)

  # finding variance
  std_1 = np.std(msk_site_1_data)

  for idx_2, row in info_df.iterrows():
    if idx_1 <= idx_2:  # Avoid duplicate or identical pairs
      continue

    site_2_id = row['site_id']

    if site_2_id not in sites:
      continue

    site_2_data, msk_site_2_data = read_TVA_or_NERC(row)

    # finding number of nans masked
    bad_2 = np.sum(msk_site_2_data.mask)

    # finding variance
    std_2 = np.std(msk_site_2_data)

    cov = ma.corrcoef(msk_site_1_data, msk_site_2_data)
    cc = cov[0, 1]
    if np.isnan(cc):
      continue

    # Compute distance between sites in km
    distance = site_distance(info_df, idx_1, idx_2)

    # Compute difference in beta
    dbeta = info_df['interpolated_beta'][idx_1] - info_df['interpolated_beta'][idx_2]

    print(f"{site_1_id}\t{site_2_id}\t{cc:+.2f}\t{distance:6.1f}\t\t{bad_1}\t{bad_2}\t{std_1:.2f}\t{std_2:.2f}\t{dbeta:.2f}")

    cc = f'{cc:.3f}'
    cc = f'[{cc}](../../../tree/main/_results/pairs/{site_1_id}_{site_2_id}.png)'

    site_1_id = site_1_id.lower().replace(' ','')
    site_2_id = site_2_id.lower().replace(' ','')
    site_1_id = f'[{site_1_id}](../../../tree/main/_processed/{site_1_id})'
    site_2_id = f'[{site_2_id}](../../../tree/main/_processed/{site_2_id})'

    rows.append([site_1_id, site_2_id, cc, distance, bad_1, bad_2, std_1, std_2, dbeta])

    # TODO:add a column in the printout of # mins

write_table(rows, out_dir)
