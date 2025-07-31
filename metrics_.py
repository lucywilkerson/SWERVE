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

def write_metrics_table(info_dict, data_all, columns, data_type=None):
    def mean_exclude_invalid(series):
        value = pd.to_numeric(series)
        valid = value[value != nan_fill]
        return np.mean(valid) if len(valid) > 0 else ''

    rows = []
    for sid in info_dict.keys():
        if data_type in info_dict[sid].keys():
            data_classes = info_dict[sid][data_type].keys()
            if 'measured' in data_classes and 'calculated' in data_classes:
                time_meas = data_all[sid][data_type]['measured'][0]['modified']['time']
                data_meas = data_all[sid][data_type]['measured'][0]['modified']['data']
                time_meas, data_meas = subset(time_meas, data_meas, start, stop)
                if data_type == 'B':
                    data_meas = np.linalg.norm(data_meas, axis=1)
                    data_meas_original = data_meas

                fill = (len(columns)-1)*[nan_fill]
                row = [sid, *fill]
                # Adding std meas
                row[1] = f"{np.nanstd(data_meas):.1f}"
                column_adjust = 2 if data_type == 'GIC' else 3
                for idx, data_source in enumerate(info_dict[sid][data_type]['calculated']):
                    if data_type == 'GIC' and data_source not in ['GMU', 'TVA']:
                        continue
                    if data_type == 'B' and data_source not in ['SWMF', 'MAGE', 'OpenGGCM']:
                        continue
                    
                    time_calc = data_all[sid][data_type]['calculated'][idx]['original']['time']
                    if data_type == 'GIC':  
                        if data_source == 'GMU':
                            # TODO: Document why GMU has multiple columns
                            data_calc = data_all[sid][data_type]['calculated'][idx]['original']['data'][:, 0]
                            data_calc = np.array(data_calc).flatten()
                            idx_adjust = 1
                        if data_source == 'TVA':
                            data_calc = data_all[sid][data_type]['calculated'][idx]['original']['data']
                            idx_adjust = 0 
                        time_calc, data_calc = subset(time_calc, data_calc, start, time_meas[-1])
                        cc = np.corrcoef(data_meas, data_calc)
                        if cc[0,1] < 0:
                            data_calc = -data_calc
                    elif data_type == 'B':
                        data_calc = data_all[sid][data_type]['calculated'][idx]['original']['data']
                        time_calc, data_calc = subset(time_calc, data_calc, start, stop)
                        data_calc = np.linalg.norm(data_calc[:,0:2], axis=1)

                        # Interpolate measured data
                        time_meas_ts = [time.mktime(t.timetuple()) for t in time_meas]
                        time_calc_ts = np.array([time.mktime(t.timetuple()) for t in time_calc])
                        # Interpolate measured data
                        data_meas = np.interp(time_calc_ts, time_meas_ts, data_meas_original)
                        idx_adjust = idx
                    
                    # Compute values for cc and pe even with nans
                    valid = ~np.isnan(data_meas) & ~np.isnan(data_calc)
                    cc = np.corrcoef(data_meas[valid], data_calc[valid])[0,1]
                    numer = np.sum((data_meas[valid]-data_calc[valid])**2)
                    denom = np.sum((data_meas[valid]-data_meas[valid].mean())**2)

                    row[2+idx_adjust] = f"{np.nanstd(data_calc):.1f}"

                    if np.sum(valid) > 1:
                        row[2+column_adjust+idx_adjust] = cc**2
                        row[2+(2*column_adjust)+idx_adjust] = 1 - numer / denom
                        for i in [2+column_adjust+idx_adjust, 2+(2*column_adjust)+idx_adjust]:
                            if np.isnan(row[i]):
                                row[i] = nan_fill
                            else:
                                row[i] = f"{row[i]:.2f}"
                    else:
                        row[2+column_adjust+idx_adjust] = nan_fill
                        row[2+(2*column_adjust)+idx_adjust] = nan_fill
                rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    logger.info(df)

    mean_row = {'site_id': 'Mean'}
    for col in df.columns:
        if col != 'site_id':
            if col.startswith('sigma'):
                mean_row[col] = f"{mean_exclude_invalid(df[col]):.1f}"
            else:
                mean_row[col] = f"{mean_exclude_invalid(df[col]):.2f}"
    df.loc[len(df)] = mean_row

    return df


columns = ['site_id', 'sigma_data', 'sigma_tva', 'sigma_gmu','cc_tva', 'cc_gmu', 'pe_tva', 'pe_gmu']
gic_df = write_metrics_table(info_dict, data_all, columns, data_type='GIC')

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
    return '' if s == nan_fill else s
formatters = {col: nan_remove for col in gic_df.columns}
print(f"Writing GIC prediction comparison tables to {fname}.{{md,tex}}")
# Apply nan_remove to each cell before writing to markdown
gic_df_md = gic_df.applymap(nan_remove)
gic_df_md.to_markdown(fname + ".md", index=False, floatfmt=".2f")
gic_df.to_latex(fname + ".tex", formatters=formatters, index=False, escape=False)


columns = ['site_id', 'sigma_data', 'sigma_swmf', 'sigma_mage', 'sigma_ggcm', 'cc_swmf', 'cc_mage', 'cc_ggcm', 'pe_swmf', 'pe_mage', 'pe_ggcm']
b_df = write_metrics_table(info_dict, data_all, columns, data_type='B')

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

fname = os.path.join(DATA_DIR, "_results", "b_table")
formatters = {col: nan_remove for col in b_df.columns}
print(f"Writing GIC prediction comparison tables to {fname}.{{md,tex}}")
# Apply nan_remove to each cell before writing to markdown
b_df_md = b_df.applymap(nan_remove)
b_df_md.to_markdown(fname + ".md", index=False, floatfmt=".2f")
b_df.to_latex(fname + ".tex", formatters=formatters, index=False, escape=False)
                    