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
columns = ['site_id', 'sigma_data', 'sigma_tva', 'sigma_gmu','cc_tva', 'cc_gmu', 'pe_tva', 'pe_gmu']
rows = []
for sid in info_dict.keys():
    if 'GIC' in info_dict[sid].keys():
        gic_types = info_dict[sid]['GIC'].keys()
        if 'measured' and 'calculated' in gic_types:
            time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
            data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
            time_meas, data_meas = subset(time_meas, data_meas, start, stop)
            row = [sid, None, None, None, None, None, None, None] #empty row to hold values
            for idx, data_source in enumerate(info_dict[sid]['GIC']['calculated']):
                if data_source not in ['GMU', 'TVA']:
                    continue

                print(f"{sid} {data_source}")

                # sigma_data
                row[1] = np.nanstd(data_meas)

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
                if data_source == 'TVA':
                    # sigma_tva
                    row[2] = np.std(data_calc)
                    # cc_tva
                    valid = ~np.isnan(data_meas) & ~np.isnan(data_calc)
                    if np.sum(valid) > 1:
                        row[4] = np.corrcoef(data_meas[valid], data_calc[valid])[0,1]
                    else:
                        row[4] = np.nan
                    # pe_tva
                    row[6] = 1-numer/denom
                elif data_source == 'GMU':
                    # sigma_gmu
                    row[3] = np.std(data_calc)
                    # cc_gmu
                    valid = ~np.isnan(data_meas) & ~np.isnan(data_calc)
                    if np.sum(valid) > 1:
                        row[5] = np.corrcoef(data_meas[valid], data_calc[valid])[0,1]
                    else:
                        row[5] = np.nan
                    # pe_gmu
                    row[7] = 1-numer/denom

            rows.append(row)

print(rows)
gic_df = pd.DataFrame(rows, columns=columns)

gic_df = gic_df.rename(columns={'site_id':'Site ID',
            'sigma_data':r'$\sigma$ [A]',
            'sigma_tva':r'$\sigma_\text{TVA}$',
            'sigma_gmu':r'$\sigma_\text{Ref}$',
            'cc_tva':r'$\text{cc}^2_\text{TVA}$',
            'cc_gmu':r'$\text{cc}^2_\text{Ref}$',
            'pe_tva':r'$\text{pe}_\text{TVA}$',
            'pe_gmu':r'$\text{pe}_\text{Ref}$'
            }
        )

fname = os.path.join(DATA_DIR, "_results", "gic_table")
print(f"Writing GIC prediction comparison tables to {fname}.{{md,tex}}")
gic_df.to_markdown(fname + ".md", index=False, floatfmt=".2f")
gic_df.to_latex(fname + ".tex", index=False, escape=False)
