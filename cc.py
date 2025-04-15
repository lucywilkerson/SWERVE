import os
import sys
import pickle
import datetime

import numpy as np
import pandas as pd
import numpy.ma as ma

from geopy.distance import geodesic

data_dir = os.path.join('..', '2024-May-Storm-data', '_processed')
out_dir = os.path.join('..', '2024-May-Storm-data', '_results')

fname = os.path.join('info', 'info.extended.csv')
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
#sites = ['10052', '10207', 'Bull Run'] # For testing

columns = ['site_1', 'site_2', 'cc', 'dist(km)', 'bad_1', 'bad_2', 'std_1', 'std_2', 'beta_diff', 'volt_diff(kV)', 'lat_diff', 'power_pool_1','power_pool_2','region_1','region_2']
print('\t'.join(columns))

def write_table(rows, rows_md, out_dir):
  # Print the results again in order of decreasing correlation coefficient
  df = pd.DataFrame(rows, columns=columns)
  df = df.sort_values(by='cc', ascending=False)
  output_fname = os.path.join(out_dir, 'cc.pkl')
  if not os.path.exists(os.path.dirname(output_fname)):
    os.makedirs(os.path.dirname(output_fname))

  print(f"Writing {output_fname}")
  with open(output_fname, 'wb') as f:
    pickle.dump(df, f)


  df = pd.DataFrame(rows_md, columns=columns)
  df = df.sort_values(by='cc', ascending=False)
  # Write the DataFrame to a markdown file
  output_fname = os.path.join(out_dir, 'cc.md')
  print(f"Writing {output_fname}")
  with open(output_fname, 'w') as f:
    f.write("See https://github.com/lucywilkerson/2024-May-Storm/blob/main/info/ for additional site information.\n\n")
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
  mod_time = site_df['modified'][0]['time'] # timestamps of 1-min avg data
  mod_data = site_df['modified'][0]['data'] # 1-min avg data
  # Crop data using subset()
  mod_time, mod_data = subset(mod_time, mod_data, start, stop)
  masked_data = ma.masked_invalid(mod_data) # 1-min data w nan values masked
  return mod_data, masked_data

def subset(time, data, start, stop):
  idx = np.logical_and(time >= start, time <= stop)
  if data.ndim == 1:
    return time[idx], data[idx]
  return time[idx], data[idx,:]

start = datetime.datetime(2024, 5, 10, 15, 0)
stop = datetime.datetime(2024, 5, 12, 6, 0)

def site_distance(df, idx_1, idx_2):
  dist = geodesic((df['geo_lat'][idx_1], df['geo_lon'][idx_1]), 
                  (df['geo_lat'][idx_2], df['geo_lon'][idx_2])).km
  return dist

def power_pool_filt(pool_1,pool_2,reg_1,reg_2):
  if pool_1 == pool_2:
    pool = pool_1
  else:
    pool = 'different'
  
  if reg_1 == reg_2:
    reg = reg_1
  else:
    reg = 'different'

  return pool, reg


rows = []
rows_md = []
for idx_1, row in info_df.iterrows():

  site_1_id = row['site_id']
  if site_1_id not in sites:
    continue

  site_1_data, msk_site_1_data = read_TVA_or_NERC(row)

  # finding number of nans masked
  bad_1 = np.sum(msk_site_1_data.mask)

  # finding variance
  std_1 = np.std(msk_site_1_data)

  # finding power pool/region
  pool_1 = row['power_pool']
  reg_1 = row['US_region']

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

    # finding power pool/region
    pool_2 = row['power_pool']
    reg_2 = row['US_region']

    cov = ma.corrcoef(msk_site_1_data, msk_site_2_data)
    cc = cov[0, 1]
    if np.isnan(cc):
      continue

    # Compute distance between sites in km
    distance = site_distance(info_df, idx_1, idx_2)

    # Compute difference in beta
    dbeta = info_df['interpolated_beta'][idx_1] - info_df['interpolated_beta'][idx_2]

    # Compute difference in voltage
    dvolt = info_df['nearest_volt'][idx_1] - info_df['nearest_volt'][idx_2]

    # Compute difference in latitude
    dlat = info_df['geo_lat'][idx_1] - info_df['geo_lat'][idx_2]

    # finding power pool/region comparison
    pool, reg = power_pool_filt(pool_1, pool_2, reg_1, reg_2)

    print(f"{site_1_id}\t{site_2_id}\t{cc:+.2f}\t{distance:6.1f}\t\t{bad_1}\t{bad_2}\t{std_1:.2f}\t{std_2:.2f}\t{dbeta:.2f}\t{dvolt:.2f}\t{dlat:.2f}\t{pool_1}\t{pool_2}\t{reg_1}\t{reg_2}")
    rows.append([site_1_id, site_2_id, cc, distance, bad_1, bad_2, std_1, std_2, dbeta, dvolt, dlat, pool_1, pool_2, reg_1, reg_2])

    # Format rows as Markdown
    site_1_id_x = site_1_id.lower().replace(' ','')
    site_2_id_x = site_2_id.lower().replace(' ','')

    cc_link = f'[{cc:.3f}](../../../tree/main/_results/pairs/{site_1_id_x}_{site_2_id_x}.png)'

    site_1_id_link = f'[{site_1_id_x}](../../../tree/main/_processed/{site_1_id_x})'
    site_2_id_link = f'[{site_2_id_x}](../../../tree/main/_processed/{site_2_id_x})'

    rows_md.append([site_1_id_link, site_2_id_link, cc_link, distance, bad_1, bad_2, std_1, std_2, dbeta, dvolt, dlat, pool_1, pool_2, reg_1, reg_2])

    # TODO:add a column in the printout of # mins

write_table(rows, rows_md, out_dir)
