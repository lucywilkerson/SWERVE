import os
import json
import pickle

import numpy as np
import numpy.ma as ma
import pandas as pd

from swerve import DATA_DIR, subset, plt_config, LOG_KWARGS, logger

logger = logger(**LOG_KWARGS)

tva_results = False #print results for TVA gic analysis
gmu_results = True #print results for GMU gic analysis
b_results = False #print results for B analysis
cc_results = False #print results for cc analysis
tva_dist = False #print distances between TVA GIC monitors and magnetometers

limits = plt_config()
start = limits['data'][0]
stop = limits['data'][1]

all_dir  = os.path.join(DATA_DIR, '_all')
all_file = os.path.join(all_dir, 'all.pkl')

def read(all_file):
  fname = os.path.join('info', 'info.json')
  with open(fname, 'r') as f:
    print(f"Reading {fname}")
    info_dict = json.load(f)

  print(f"Reading {all_file}")
  with open(all_file, 'rb') as f:
    data_all = pickle.load(f)

  return info_dict, data_all


info_dict, data_all = read(all_file)

TVA_sites = ['Bull Run', 'Montgomery', 'Union', 'Widows Creek']
ccs = []
pes = []
number_gmu_sites = 0
columns = ['site_id', 'model', 'sigma_data', 'sigma_model', 'cc', 'pe']
rows = []
for sid in info_dict.keys():
    if 'GIC' in info_dict[sid].keys():
        gic_types = info_dict[sid]['GIC'].keys()
        if 'measured' and 'calculated' in gic_types:
            time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
            data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
            time_meas, data_meas = subset(time_meas, data_meas, start, stop)

            for idx, data_source in enumerate(info_dict[sid]['GIC']['calculated']):
                if data_source not in ['GMU', 'TVA']:
                    continue

                print(f"{sid} {data_source}")
                row = [sid, None, None, None, None, None]

                # model
                row[1] = data_source

                # sigma_data
                row[2] = np.std(data_meas)

                time_calc = data_all[sid]['GIC']['calculated'][idx]['original']['time']
                if data_source == 'GMU':
                    # TODO: Document why GMU has multiple columns
                    data_calc = data_all[sid]['GIC']['calculated'][idx]['original']['data'][:, 0]
                    data_calc = np.array(data_calc).flatten()
                if data_source == 'TVA':
                    data_calc = data_all[sid]['GIC']['calculated'][idx]['original']['data']

                time_calc, data_calc = subset(time_calc, data_calc, start, time_meas[-1])
                cc = np.corrcoef(data_meas, data_calc)
                if cc[0,1] < 0:
                    data_calc = -data_calc
                numer = np.sum((data_meas-data_calc)**2)
                denom = np.sum((data_meas-data_meas.mean())**2)

                # sigma_model
                row[3] = np.std(data_calc)
                # cc
                row[4] = np.corrcoef(data_meas, data_calc)[0,1]
                # pe
                row[5] = 1-numer/denom

                rows.append(row)

print(rows)
gic_df = pd.DataFrame(rows, columns=columns)
fname = os.path.join(DATA_DIR, "_results", "gic_table")
print(f"Writing GIC prediction comparison tables to {fname}.{{md,tex}}")
gic_df.to_markdown(fname + ".md", index=False, floatfmt=".2f")
gic_df.to_latex(fname + ".tex", index=False, escape=False)

if False:
        df.loc[len(df)] = {
            'Site ID': sid,
            r'$\sigma$ [A]': f"{np.std(data_meas):.2f}",
            r'$\sigma_\text{TVA}$': f"{np.std(data_calcs[0]):.2f}",
            r'$\sigma_\text{Ref}$': f"{np.std(data_calcs[1]):.2f}",
            r'$\text{cc}^2_\text{TVA}$': f"{cc[0]**2:.2f}",
            r'$\text{cc}^2_\text{Ref}$': f"{cc[1]**2:.2f}",
            r'$\text{pe}_\text{TVA}$': f"{pe[0]:.2f}",
            r'$\text{pe}_\text{Ref}$': f"{pe[1]:.2f}"
        }