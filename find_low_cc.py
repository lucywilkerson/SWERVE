import os
import pandas as pd
import pickle

import numpy as np


data_dir = os.path.join('..', '2024-AGU-data')
out_dir = os.path.join('..', '2024-AGU-data', '_map')

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

# Read in cc data
pkl_file = os.path.join(data_dir, '_results', 'cc.pkl')
with open(pkl_file, 'rb') as file:
  print(f"Reading {pkl_file}")
  cc_rows = pickle.load(file)
cc_df = pd.DataFrame(cc_rows)
cc_df.reset_index(drop=True, inplace=True)


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
        if n_site == 55:
            print(site_id)
    for idx_1, row in info_df.iterrows():
        site_1_id = row['site_id']
        is_site_bad(site_1_id, cc_df)

cc_max = 0.3
site_filt(info_df, cc_df, cc_max)
