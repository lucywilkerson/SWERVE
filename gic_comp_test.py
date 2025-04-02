import os
import json
import numpy
import pickle
import shutil
import datetime
import time

import numpy as np
import pandas as pd
import numpy.ma as ma
 

from datetick import datetick 


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600


fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)
print(info_df)
site_df = info_df[info_df['data_source'].str.contains('GMU', na=False)]
site_df.reset_index(drop=True, inplace=True)

data_dir = os.path.join('..', '2024-AGU-data', 'dennies_gic_comparison')

for site in site_df['site_id'].tolist():
    fname = os.path.join(data_dir, 'tva', f'site_{site}.csv')
    if not os.path.exists(fname):
        fname = os.path.join(data_dir, 'nerc', f'site_{site}.csv')
    if not os.path.exists(fname):
        info_df.loc[info_df['site_id'] == site, 'error'] = "no GIC comparison file"
        continue
    print(f"Reading {fname}")
    df = pd.read_csv(fname)
    data_calc = df['Sim GIC (Median)']
    data_meas = df['site_1_gic']
    time_meas = pd.to_datetime(df['timestamp'])
    calc_label = df['substation'][0]
    #plt.plot(time_meas, data_calc, label=f'Sim GIC (Median) @ {calc_label}')
    #print(df.keys())
    meas_label = df['site_1_device'][0]
    #plt.plot(time_meas, data_meas, label=meas_label)
    #datetick()
    #plt.legend()
    #plt.show()
    #plt.close()


output_fname = os.path.join('info', 'info.csv')
info_df.to_csv(output_fname, index=False)
print(f"Saved updated info_df to {output_fname}")

