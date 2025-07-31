import os
import json
import pickle

import numpy as np
import numpy.ma as ma
import pandas as pd

import time

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
nan_fill = -99999

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


columns = ['site_id', 'sigma_data', 'sigma_tva', 'sigma_gmu','cc_tva', 'cc_gmu', 'pe_tva', 'pe_gmu']
rows = []
for sid in info_dict.keys():
    if 'GIC' in info_dict[sid].keys():
        gic_types = info_dict[sid]['GIC'].keys()
        if 'measured' and 'calculated' in gic_types:
            time_meas = data_all[sid]['GIC']['measured'][0]['modified']['time']
            data_meas = data_all[sid]['GIC']['measured'][0]['modified']['data']
            time_meas, data_meas = subset(time_meas, data_meas, start, stop)

            fill = 7*[nan_fill]
            row = [sid, *fill]
            for idx, data_source in enumerate(info_dict[sid]['GIC']['calculated']):
                if data_source not in ['GMU', 'TVA']:
                    continue

                # sigma_data
                row[1] = f"{np.nanstd(data_meas):.1f}"

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
                # Compute values for cc and pe even with nans
                valid = ~np.isnan(data_meas) & ~np.isnan(data_calc)
                cc = np.corrcoef(data_meas[valid], data_calc[valid])[0,1]
                numer = np.sum((data_meas[valid]-data_calc[valid])**2)
                denom = np.sum((data_meas[valid]-data_meas[valid].mean())**2)
                if data_source == 'TVA':
                    # sigma_tva
                    row[2] = f"{np.nanstd(data_calc):.1f}"
                    # cc_tva and pe_tva
                    if np.sum(valid) > 1:
                        row[4] = f"{cc**2:.2f}"
                        row[6] = f"{1-numer/denom:.2f}"
                    else:
                        row[4] = nan_fill
                        row[6] = nan_fill
                    
                elif data_source == 'GMU':
                    # sigma_gmu
                    row[3] = f"{np.nanstd(data_calc):.1f}"
                    # cc_gmu and pe_gmu
                    if np.sum(valid) > 1:
                        row[5] = cc**2
                        row[7] = 1 - numer / denom
                        for i in [5, 7]:
                            if np.isnan(row[i]):
                                row[i] = nan_fill
                            else:
                                row[i] = f"{row[i]:.2f}"
                    else:
                        row[5] = nan_fill
                        row[7] = nan_fill
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

def mean_exclude_invalid(series):
    value = pd.to_numeric(series)
    valid = value[value != nan_fill]
    return np.mean(valid) if len(valid) > 0 else ''

mean_row = {'Site ID': 'Mean'}
for col in gic_df.columns:
    if col != 'Site ID':
        if col.startswith(r'$\sigma'):
            mean_row[col] = f"{mean_exclude_invalid(gic_df[col]):.1f}"
        else:
            mean_row[col] = f"{mean_exclude_invalid(gic_df[col]):.2f}"
gic_df.loc[len(gic_df)] = mean_row

fname = os.path.join(DATA_DIR, "_results", "gic_table")
def nan_remove(s):
    print(s)
    return '' if s == nan_fill else s
formatters = {col: nan_remove for col in gic_df.columns}
print(f"Writing GIC prediction comparison tables to {fname}.{{md,tex}}")
# Apply nan_remove to each cell before writing to markdown
gic_df_md = gic_df.applymap(nan_remove)
gic_df_md.to_markdown(fname + ".md", index=False, floatfmt=".2f")
gic_df.to_latex(fname + ".tex", formatters=formatters, index=False, escape=False)


columns = ['site_id', 'sigma_data', 'sigma_swmf', 'sigma_mage', 'sigma_ggcm', 'cc_swmf', 'cc_mage', 'cc_ggcm', 'pe_swmf', 'pe_mage', 'pe_ggcm']
rows = []
for sid in info_dict.keys():
    if 'B' in info_dict[sid].keys():
        b_types = info_dict[sid]['B'].keys()
        if 'measured' in b_types and 'calculated' in b_types:
            time_meas = data_all[sid]['B']['measured'][0]['modified']['time']
            data_meas = data_all[sid]['B']['measured'][0]['modified']['data']
            time_meas, data_meas = subset(time_meas, data_meas, start, stop)
            data_meas = np.linalg.norm(data_meas, axis=1)

            fill = 10*[nan_fill]
            row = [sid, *fill]
            for idx, data_source in enumerate(info_dict[sid]['B']['calculated']):
                if data_source not in ['SWMF', 'MAGE', 'OpenGGCM']:
                    continue

                # Adding std meas
                row[1] = f"{np.nanstd(data_meas):.1f}"

                time_calc = data_all[sid]['B']['calculated'][idx]['original']['time']
                data_calc = data_all[sid]['B']['calculated'][idx]['original']['data']
                time_calc, data_calc = subset(time_calc, data_calc, start, stop)
                data_calc = np.linalg.norm(data_calc[:,0:2], axis=1)

                # Interpolate measured data
                time_meas_ts = [time.mktime(t.timetuple()) for t in time_meas]
                time_calc_ts = np.array([time.mktime(t.timetuple()) for t in time_calc])
                # Interpolate measured data
                data_interp = np.interp(time_calc_ts, time_meas_ts, data_meas)
                # Calculate cc and pe excluding nans
                valid = ~np.isnan(data_interp) & ~np.isnan(data_calc)
                cc = np.corrcoef(data_interp[valid], data_calc[valid])[0,1]
                numer = np.sum((data_interp[valid]-data_calc[valid])**2)
                denom = np.sum((data_interp[valid]-data_interp[valid].mean())**2)
                # Adding std calc
                row[2+idx] = f"{np.nanstd(data_calc):.1f}"
                # Adding cc and pe
                if np.sum(valid) > 1:
                    row[5+idx] = f"{cc**2:.2f}"
                    row[8+idx] = f"{1-numer/denom:.2f}"
                else:
                    row[5+idx] = nan_fill
                    row[8+idx] = nan_fill

            rows.append(row)

b_df = pd.DataFrame(rows, columns=columns)

logger.info(b_df)

b_df = b_df.rename(columns={'site_id':'Site ID',
            'sigma_data':r'$\sigma$ [nT]',
            'sigma_swmf':r'$\sigma_\text{SWMF}$',
            'sigma_mage':r'$\sigma_\text{MAGE}$',
            'sigma_ggcm':r'$\sigma_\text{GGCM}$',
            'cc_swmf':r'$\text{cc}^2_\text{SWMF}$',
            'cc_mage':r'$\text{cc}^2_\text{MAGE}$',
            'cc_ggcm':r'$\text{cc}^2_\text{GGCM}$',
            'pe_swmf':r'$\text{pe}_\text{SWMF}$',
            'pe_mage':r'$\text{pe}_\text{MAGE}$',
            'pe_ggcm':r'$\text{pe}_\text{GGCM}$'
            }
        )

mean_row = {'Site ID': 'Mean'}
for col in b_df.columns:
    if col != 'Site ID':
        if col.startswith(r'$\sigma'):
            mean_row[col] = f"{mean_exclude_invalid(b_df[col]):.1f}"
        else:
            mean_row[col] = f"{mean_exclude_invalid(b_df[col]):.2f}"
b_df.loc[len(b_df)] = mean_row

fname = os.path.join(DATA_DIR, "_results", "b_table")
formatters = {col: nan_remove for col in b_df.columns}
print(f"Writing GIC prediction comparison tables to {fname}.{{md,tex}}")
# Apply nan_remove to each cell before writing to markdown
b_df_md = b_df.applymap(nan_remove)
b_df_md.to_markdown(fname + ".md", index=False, floatfmt=".2f")
b_df.to_latex(fname + ".tex", formatters=formatters, index=False, escape=False)