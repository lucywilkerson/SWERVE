import os
import numpy as np
import pandas as pd

data_dir = os.path.join('..', '2024-May-Storm-data')
base_dir = os.path.join(data_dir, '_processed')

fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)
info_df = info_df[info_df['data_source'] != 'GMU']
meas_df = info_df[info_df['data_class'] == 'measured']

data_dir = os.path.join(data_dir, 'dennies_gic_comparison')
fname = os.path.join(data_dir, 'gic_mean_df_1.csv')
print(f"Reading {fname}")
site_df = pd.read_csv(fname)

# finding associated tva/nerc sites for all simulation sites 
output_data = []
for site in site_df['sub_id'].unique():
  tva_df = None
  nerc_df = None
  tva_site = None
  nerc_site = None

  tva_fname = os.path.join(data_dir, 'tva', f'site_{site}.csv')
  if os.path.exists(tva_fname):
    tva_df = pd.read_csv(tva_fname)
    tva_site = tva_df['site_1_device'][0]
    if tva_site == 'Widows Creek 2':
      tva_site = 'Widows Creek'
    if tva_site == 'Paradise':
      tva_site = 'Paradise 3'
  
  nerc_fname = os.path.join(data_dir, 'nerc', f'site_{site}.csv')
  if os.path.exists(nerc_fname):
    nerc_df = pd.read_csv(nerc_fname)
    nerc_site = f'{nerc_df['site_1_device'][0]}'
  
  if tva_df is not None or nerc_df is not None:
    sim_lat = None
    sim_lon = None
    tva_lat = None
    tva_lon = None
    tva_dist = None
    nerc_lat = None
    nerc_lon = None
    nerc_dist = None

    if tva_df is not None:
      sim_lat = tva_df['sub lat'][0]
      sim_lon = tva_df['sub lon'][0]
      tva_lat = meas_df.loc[meas_df['site_id'] == tva_site, 'geo_lat'].values[0]
      tva_lon = meas_df.loc[meas_df['site_id'] == tva_site, 'geo_lon'].values[0]
      tva_dist = tva_df['site_1_Distance'][0]

    if nerc_df is not None:
      if sim_lat is None:  # If not already set by TVA
        sim_lat = nerc_df['sub lat'][0]
        sim_lon = nerc_df['sub lon'][0]
      nerc_lat = meas_df.loc[meas_df['site_id'] == nerc_site, 'geo_lat'].values[0]
      nerc_lon = meas_df.loc[meas_df['site_id'] == nerc_site, 'geo_lon'].values[0]
      nerc_dist = nerc_df['site_1_Distance'][0]

    output_data.append({
      'sim_site': site,
      'sim_lat': sim_lat,
      'sim_lon': sim_lon,
      'tva_site': tva_site,
      'tva_lat': tva_lat,
      'tva_lon': tva_lon,
      'tva_dist': tva_dist,
      'nerc_site': nerc_site,
      'nerc_lat': nerc_lat,
      'nerc_lon': nerc_lon,
      'nerc_dist': nerc_dist
    })

# output data to df
output_df = pd.DataFrame(output_data)

# adding GMU simulation sites to info.csv

def add_info(info_df, output_df, data_source=['tva','nerc']):
  for site_type in data_source:
    for site in output_df[f'{site_type}_site'].unique():
      df = output_df[output_df[f'{site_type}_site']==site]
      if not df.empty and f'{site_type}_dist' in df.columns:
        min_dist_row = df.loc[df[f'{site_type}_dist'].idxmin()]
      else:
        continue

      new_row = pd.DataFrame([{
        'site_id': min_dist_row[f'{site_type}_site'],
        'geo_lat': min_dist_row[f'{site_type}_lat'],
        'geo_lon': min_dist_row[f'{site_type}_lon'],
        'data_type': 'GIC',
        'data_class': 'calculated',
        'data_source': 'GMU',
        'error': np.nan
      }])
      info_df = pd.concat([info_df, new_row], ignore_index=True)
  return info_df
  

new_info_df = add_info(info_df, output_df)

#output_fname = os.path.join('info', 'info.csv')
#new_info_df.to_csv(output_fname, index=False)
#print(f"Saved updated info table to {output_fname}")

#####  RUN BETA.PY, GRID_REGION.PY, AND VOLTAGE.PY BEFORE CONTINUING!!  #####

fname = os.path.join('info', 'info.extended.csv')
print(f"Reading {fname}")
info_ex_df = pd.read_csv(fname)

# add new column based on the nearest simulation site
info_ex_df['nearest_sim_site'] = np.nan  
def add_nearest_sim_site(info_df, output_df, data_source=['tva', 'nerc']):
  for site_type in data_source:
    for site in output_df[f'{site_type}_site'].unique():
      df = output_df[output_df[f'{site_type}_site'] == site]
      if not df.empty and f'{site_type}_dist' in df.columns:
        min_dist_row = df.loc[df[f'{site_type}_dist'].idxmin()]
      else:
        continue

      mask = (info_df['data_source'] == 'GMU') & (info_df['site_id'] == min_dist_row[f'{site_type}_site'])
      info_df.loc[mask, 'nearest_sim_site'] = f'{min_dist_row['sim_site']}'
  return info_df

info_ex_df = add_nearest_sim_site(info_ex_df, output_df)

#out_fname = os.path.join('info', 'info.extended.csv')
#info_ex_df.to_csv(out_fname, index=False)
#print(f"Saving updated {out_fname}")