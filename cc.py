import os
import pickle
import datetime

import numpy as np
import pandas as pd
import json

from geopy.distance import geodesic

data_dir = os.path.join('..', '2024-May-Storm-data', '_processed')
out_dir = os.path.join('..', '2024-May-Storm-data', '_results')
all_dir  = os.path.join('..', '2024-May-Storm-data', '_all')
all_file = os.path.join(all_dir, 'all.pkl')

def read(all_file, sid=None):
  fname = os.path.join('info', 'info.json')
  with open(fname, 'r') as f:
    print(f"Reading {fname}")
    info_dict = json.load(f)

  info_df = pd.read_csv(os.path.join('info', 'info.csv'))

  fname = os.path.join('info', 'plot.json')
  with open(fname, 'r') as f:
    print(f"Reading {fname}")
    plot_cfg = json.load(f)

  print(f"Reading {all_file}")
  with open(all_file, 'rb') as f:
    data = pickle.load(f)

  return info_dict, info_df, data, plot_cfg

info_dict, info_df, data_all, plot_info = read(all_file)

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

columns = ['site_1', 'site_2', 'cc', 'dist(km)', 'bad_1', 'bad_2', 'std_1', 'std_2', 'beta_diff', 'volt_diff(kV)', 'lat_diff', 'power_pool_1','power_pool_2','region_1','region_2','peak_xcorr','peak_xcorr_lag(min)']
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

  time_meas_1 = data_all[site_1_id]['GIC']['measured'][0]['modified']['time']
  data_meas_1 = data_all[site_1_id]['GIC']['measured'][0]['modified']['data']
  time_meas_1, data_meas_1 = subset(time_meas_1, data_meas_1, start, stop)

  # finding number of nans
  bad_1 = np.isnan(data_meas_1).sum()

  # finding variance
  std_1 = np.std(data_meas_1[~np.isnan(data_meas_1)])

  # finding power pool/region
  pool_1 = row['power_pool']
  reg_1 = row['US_region']

  for idx_2, row in info_df.iterrows():
    if idx_1 <= idx_2:  # Avoid duplicate or identical pairs
      continue

    site_2_id = row['site_id']

    if site_2_id not in sites:
      continue

    time_meas_2 = data_all[site_2_id]['GIC']['measured'][0]['modified']['time']
    data_meas_2 = data_all[site_2_id]['GIC']['measured'][0]['modified']['data']
    time_meas_2, data_meas_2 = subset(time_meas_2, data_meas_2, start, stop)

    # finding number of nans
    bad_2 = np.isnan(data_meas_2).sum()

    # finding variance
    std_2 = np.std(data_meas_2[~np.isnan(data_meas_2)])

    # finding power pool/region
    pool_2 = row['power_pool']
    reg_2 = row['US_region']

    valid_mask = ~np.isnan(data_meas_1) & ~np.isnan(data_meas_2)
    cov = np.corrcoef(data_meas_1[valid_mask], data_meas_2[valid_mask])
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

    # finding peak cross correlation
    if cc < 0:
      data_meas_1 = -data_meas_1
    lags = np.arange(-60, 61, 1) 
    cross_corr = [np.corrcoef(data_meas_2[~np.isnan(data_meas_2) & ~np.isnan(np.roll(data_meas_1, lag))], 
                            np.roll(data_meas_1, lag)[~np.isnan(data_meas_2) & ~np.isnan(np.roll(data_meas_1, lag))])[0, 1] for lag in lags]
    max_xcorr = max(cross_corr)
    max_lag = lags[cross_corr.index(max_xcorr)]

    print(f"{site_1_id}\t{site_2_id}\t{cc:+.2f}\t{distance:6.1f}\t\t{bad_1}\t{bad_2}\t{std_1:.2f}\t{std_2:.2f}\t{dbeta:.2f}\t{dvolt:.2f}\t{dlat:.2f}\t{pool_1}\t{pool_2}\t{reg_1}\t{reg_2}\t{max_xcorr:.2f}\t{max_lag}")
    rows.append([site_1_id, site_2_id, cc, distance, bad_1, bad_2, std_1, std_2, dbeta, dvolt, dlat, pool_1, pool_2, reg_1, reg_2, max_xcorr, max_lag])

    # Format rows as Markdown
    site_1_id_x = site_1_id.lower().replace(' ','')
    site_2_id_x = site_2_id.lower().replace(' ','')

    cc_link = f'[{cc:.3f}](../../../tree/main/_results/pairs/{site_2_id_x}_{site_1_id_x}.png)'

    site_1_id_link = f'[{site_1_id_x}](../../../tree/main/_processed/{site_1_id_x})'
    site_2_id_link = f'[{site_2_id_x}](../../../tree/main/_processed/{site_2_id_x})'

    rows_md.append([site_1_id_link, site_2_id_link, cc_link, distance, bad_1, bad_2, std_1, std_2, dbeta, dvolt, dlat, pool_1, pool_2, reg_1, reg_2, max_xcorr, max_lag])

    # TODO:add a column in the printout of # mins

write_table(rows, rows_md, out_dir)

#################################################################################################################
# add column for minimum mean |cc| for each site

# Read in cc data
pkl_file = os.path.join(out_dir, 'cc.pkl')
with open(pkl_file, 'rb') as file:
  print(f"Reading {pkl_file}")
  cc_rows = pickle.load(file)
cc_df = pd.DataFrame(cc_rows)
cc_df.reset_index(drop=True, inplace=True)

# defining avg |cc| for each site
for idx_1, row in info_df.iterrows():
    site_1_id = row['site_id']
    site_cc = []
    for idx_2, row in cc_df.iterrows():
        if row['site_1'] == site_1_id:
            site_2_id = row['site_2']
        elif row['site_2'] == site_1_id:
            site_2_id = row['site_1']
        else:
            continue
        cc = np.abs(row['cc'])
        site_cc.append(cc)
    avg_cc = np.mean(site_cc)
    info_df.loc[info_df['site_id'] == site_1_id, 'avg_cc'] = avg_cc #adding mean cc to info_df

# adding min(avg_cc) to cc_df
for idx, row in cc_df.iterrows():
    site_1_id = row['site_1']
    site_2_id = row['site_2']
    avg_cc_1 = info_df.loc[info_df['site_id'] == site_1_id, 'avg_cc'].values[0]
    avg_cc_2 = info_df.loc[info_df['site_id'] == site_2_id, 'avg_cc'].values[0]
    cc_df.loc[idx, 'min_avg_cc'] = min(avg_cc_1, avg_cc_2)


output_fname = os.path.join(out_dir, 'cc.pkl')
if not os.path.exists(os.path.dirname(output_fname)):
    os.makedirs(os.path.dirname(output_fname))
print(f"Writing {output_fname}")
with open(output_fname, 'wb') as f:
    pickle.dump(cc_df, f)

############################################################################################################
# finding sites with no paring beyond a given cc limit
find_lowest_cc = False

def site_filt(info_df, cc_df, cc_lim):
    print(f'Sites with no pairing |cc| > {cc_lim}:')
    def is_site_bad(site_id, cc_df):
        n_site = 0
        for idx, row in cc_df.iterrows():
            if row['site_1'] == site_id:
                site_2_id = row['site_2']
            elif row['site_2'] == site_id:
                site_2_id = row['site_1']
            else:
                continue
            cc = np.abs(row['cc'])
            if cc > cc_lim:
                break
            n_site += 1
        if n_site == len(info_df)-1:
            print(site_id)
    for idx_1, row in info_df.iterrows():
        site_1_id = row['site_id']
        is_site_bad(site_1_id, cc_df)

if find_lowest_cc:
  #reading in info.extended.csv
  fname = os.path.join('info', 'info.extended.csv')
  print(f"Reading {fname}")
  df = pd.read_csv(fname).set_index('site_id')
  info_df = pd.read_csv(fname)

  # Filter out sites with error message
  info_df = info_df[~info_df['error'].str.contains('', na=False)]
  # TODO: Print number of GIC sites removed due to error and how many kept.
  # Remove rows that don't have data_type = GIC and data_class = measured
  info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
  info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
  info_df.reset_index(drop=True, inplace=True)

  cc_max = 0.3
  site_filt(info_df, cc_df, cc_max)