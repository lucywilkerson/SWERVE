## To be added to site_stats_summary.py

import os
import numpy as np
import pandas as pd

from swerve import config, site_read, site_stats, read_info_dict

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

DATA_DIR = os.path.join(CONFIG['dirs']['data'])
paper_dir = os.path.join(CONFIG['dirs']['paper'])
all_dir  = os.path.join(DATA_DIR, '_all')
all_file = os.path.join(all_dir, 'all.pkl')

nan_fill = -99999

info_dict = read_info_dict()

TVA_sites = ['Bull Run', 'Montgomery', 'Union', 'Widows Creek']

def write_metrics_table(info_dict, column_names, data_type):
    from swerve import fix_latex
    def mean_exclude_invalid(series): #excludes nan_fill values from mean calculation
        value = pd.to_numeric(series)
        valid = value[value != nan_fill]
        return np.mean(valid) if len(valid) > 0 else ''
    
    def nan_remove(s): #removes nan_fill values
        print(s)
        return '' if s == nan_fill else s

    columns = list(column_names.keys()) #names of columns
    rows = []
    for sid in info_dict.keys():
        if sid.startswith('test'): #skipping test sites
            continue
        if data_type in info_dict[sid].keys():
            data_classes = info_dict[sid][data_type].keys()
            if 'measured' in data_classes and 'calculated' in data_classes:
                #Reading site data
                data = site_read(sid, data_types=data_type, logger=logger)
                stats = site_stats(sid, data, data_types=data_type, logger=logger)
                # Setting up row for site
                row = {col: nan_fill for col in columns}
                row['site_id'] = sid
                for data_source in info_dict[sid][data_type]['measured']:
                    if 'stats' in stats[f'{data_type}/measured/{data_source}']:
                        # Adding std meas
                        data_std = stats[f'{data_type}/measured/{data_source}']['stats']['std'][0]
                        row['sigma_data'] = f"{data_std:.1f}"

                for data_source in info_dict[sid][data_type]['calculated']:
                    if data_type == 'GIC' and data_source not in ['GMU', 'TVA']:
                        continue
                    if data_type == 'B' and data_source not in ['SWMF', 'MAGE', 'OpenGGCM']:
                        continue

                    # Calculated std
                    calc_std = stats[f'{data_type}/calculated/{data_source}']['stats']['std'][0]
                    # Save to row
                    row[f'sigma_{data_source.lower()}'] = f"{calc_std:.1f}"
                    # Calculated cc and pe
                    if 'metrics' in stats[f'{data_type}/calculated/{data_source}']:
                        calc_cc = stats[f'{data_type}/calculated/{data_source}']['metrics']['cc'][-1]
                        calc_pe = stats[f'{data_type}/calculated/{data_source}']['metrics']['pe'][-1]
                        # Save to row
                        row[f'cc_{data_source.lower()}'] = calc_cc**2
                        row[f'pe_{data_source.lower()}'] = calc_pe
                        for i in [f'cc_{data_source.lower()}', f'pe_{data_source.lower()}']:
                            if np.isnan(row[i]):
                                row[i] = nan_fill #fill with nan_fill if cc is nan
                            else:
                                row[i] = f"{row[i]:.2f}"
                    else: #fill with nan_fill if not enough valid data
                        row[f'cc_{data_source.lower()}'] = nan_fill
                        row[f'pe_{data_source.lower()}'] = nan_fill
                rows.append(row)
    df = pd.DataFrame(rows, columns=columns) #create dataframe
    logger.info(df)
    # Add row of mean values
    mean_row = {'site_id': 'Mean'}
    for col in df.columns:
        if col != 'site_id':
            if col.startswith('sigma'):
                mean_row[col] = f"{mean_exclude_invalid(df[col]):.1f}"
            else:
                mean_row[col] = f"{mean_exclude_invalid(df[col]):.2f}"
    df.loc[len(df)] = mean_row
    #rename columns with LaTex
    df = df.rename(columns=column_names)
    # Format and save df as .md and .tex files
    fname = os.path.join(DATA_DIR, "_results", f"{data_type.lower()}_table")
    formatters = {col: nan_remove for col in df.columns}
    logger.info(f"Writing {data_type} prediction comparison tables to {fname}.{{md,tex}}")
    # Apply nan_remove to each cell before writing to markdown
    df_md = df.applymap(nan_remove)
    df_md.to_markdown(fname + ".md", index=False, floatfmt=".2f")
    df_tex = fix_latex(df, data_type, formatters=formatters)
    with open(fname + ".tex", "w") as f:
        f.write(df_tex)
    fname = os.path.join(paper_dir, "figures", "_results", f"{data_type.lower()}_table")
    with open(fname + ".tex", "w") as f:
        f.write(df_tex)

    return df


# Define mapping from original columns to renamed columns
gic_column_names = {
    'site_id': 'Site ID',
    'sigma_data': r'$\sigma$ [A]',
    'sigma_tva': r'$\sigma_\text{TVA}$',
    'sigma_gmu': r'$\sigma_\text{Ref}$',
    'cc_tva': r'$\text{cc}^2_\text{TVA}$',
    'cc_gmu': r'$\text{cc}^2_\text{Ref}$',
    'pe_tva': r'$\text{pe}_\text{TVA}$',
    'pe_gmu': r'$\text{pe}_\text{Ref}$'
}

gic_df = write_metrics_table(info_dict, gic_column_names, data_type='GIC')


b_column_names = {'site_id':'Site ID',
            'sigma_data':r'$\sigma$ [nT]',
            'sigma_swmf':r'$\sigma_\text{SWMF}$',
            'sigma_mage':r'$\sigma_\text{MAGE}$',
            'sigma_openggcm':r'$\sigma_\text{GGCM}$',
            'cc_swmf':r'$\text{cc}^2_\text{SWMF}$',
            'cc_mage':r'$\text{cc}^2_\text{MAGE}$',
            'cc_openggcm':r'$\text{cc}^2_\text{GGCM}$',
            'pe_swmf':r'$\text{pe}_\text{SWMF}$',
            'pe_mage':r'$\text{pe}_\text{MAGE}$',
            'pe_openggcm':r'$\text{pe}_\text{GGCM}$'
            }

b_df = write_metrics_table(info_dict, b_column_names, data_type='B')  
