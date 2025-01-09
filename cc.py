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


sites = info_df['site_id']
sites = ['10052', '10064'] # For testing
print("site_1\tsite_2\tcc")
rows = []
for idx_1, site_1 in enumerate(sites):
  for idx_2, site_2 in enumerate(sites):
    # TODO: 1/2 ccs are duplicate b/c cc(a,b) = cc(b,a). Figure out how to modify the loop to avoid this.
    if idx_1 == idx_2:
      continue
    #TODO: cc = ...
    cc = 1.0
    print(f"{site_1}\t{site_2}\t{cc}")
    rows.append([site_1, site_2, cc])


# TODO: Print the results again in order of decreasing correlation coefficient

