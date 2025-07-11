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

            fill = 7*[-99999]
            row = [sid, *fill]
            for idx, data_source in enumerate(info_dict[sid]['GIC']['calculated']):
                if data_source not in ['GMU', 'TVA']:
                    continue

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
                    # Compute correlation even if data_meas has nans
                    valid = ~np.isnan(data_meas) & ~np.isnan(data_calc)
                    if np.sum(valid) > 1:
                        row[4] = np.corrcoef(data_meas[valid], data_calc[valid])[0,1]**2
                    else:
                        row[4] = -99999
                    # pe_tva
                    row[6] = 1-numer/denom
                elif data_source == 'GMU':
                    # sigma_gmu
                    row[3] = np.std(data_calc)
                    # cc_gmu
                    valid = ~np.isnan(data_meas) & ~np.isnan(data_calc)
                    if np.sum(valid) > 1:
                        cc_gmu = np.corrcoef(data_meas[valid], data_calc[valid])[0,1]**2
                        if np.isnan(cc_gmu):
                            cc_gmu = -99999
                    else:
                        cc_gmu = -99999
                    row[5] = cc_gmu
                    # pe_gmu
                    pe_gmu = 1 - numer / denom
                    if np.isnan(pe_gmu):
                        pe_gmu = -99999
                    row[7] = pe_gmu

            rows.append(row)

gic_df = pd.DataFrame(rows, columns=columns)

logger.info(gic_df)

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
def nan_remove(s):
    print(s)
    return '' if s == -99999 else s
formatters = {col: nan_remove for col in gic_df.columns}
print(f"Writing GIC prediction comparison tables to {fname}.{{md,tex}}")
# Apply nan_remove to each cell before writing to markdown
gic_df_md = gic_df.applymap(nan_remove)
gic_df_md.to_markdown(fname + ".md", index=False, floatfmt=".2f")
gic_df.to_latex(fname + ".tex", formatters=formatters, index=False, escape=False)
