import os
import numpy as np
import pandas as pd


def site_stats_summary(stats, data_types=None, logger=None, nan_fill=-99999):

    # Explain nan_fill

    from swerve import config, read_info_dict, fix_latex

    CONFIG = config()

    DATA_DIR = os.path.join(CONFIG['dirs']['data'])
    paper_dir = os.path.join(CONFIG['dirs']['paper'])

    info_dict = read_info_dict()

    if data_types is None:
        data_types = ['GIC', 'B']

    for data_type in data_types:
        n_neg_cc = 0
        n_sites = 0
        column_names = _column_names(data_type) #get column names
        columns = list(column_names.keys()) #names of columns
        # Set indexing
        idx = 0 if data_type == 'GIC' else 3
        rows = []
        for sid in stats.keys():
            if sid.startswith('test'): #skipping test sites
                continue
            if data_type not in info_dict[sid].keys():
                continue # Skip if data_type not in info_dict

            data_classes = info_dict[sid][data_type].keys()
            if not ('measured' in data_classes and 'calculated' in data_classes):
                continue

            # Reading site data
            sid_stats = stats[sid]
            # Setting up row for site
            row = {col: nan_fill for col in columns}
            row['site_id'] = sid
            for data_source in info_dict[sid][data_type]['measured']:
                path = f'{data_type}/measured/{data_source}'
                if 'stats' in sid_stats[path]:
                    # Adding std meas
                    data_std = stats[sid][path]['stats']['std'][idx]
                    row['sigma_data'] = f"{data_std:.1f}"

            for data_source in info_dict[sid][data_type]['calculated']:
                n_sites += 1
                
                if data_type == 'GIC' and data_source not in ['GMU', 'TVA']:
                    continue
                if data_type == 'B' and data_source not in ['SWMF', 'MAGE', 'OpenGGCM']:
                    continue

                # Calculated std
                path = f'{data_type}/calculated/{data_source}'
                calc_std = sid_stats[path]['stats']['std'][idx]
                # Save to row
                data_source_lower = data_source.lower()
                row[f'sigma_{data_source_lower}'] = f"{calc_std:.1f}"
                # Calculated cc and pe
                if 'metrics' in sid_stats[path]:
                    calc_cc = sid_stats[path]['metrics']['cc'][idx]
                    calc_pe = sid_stats[path]['metrics']['pe'][idx]
                    # Save to row
                    row[f'cc_{data_source_lower}'] = calc_cc**2
                    row[f'pe_{data_source_lower}'] = calc_pe
                    for i in [f'cc_{data_source_lower}', f'pe_{data_source_lower}']:
                        if np.isnan(row[i]):
                            row[i] = nan_fill # fill with nan_fill if cc is nan
                        elif i.startswith('pe') and calc_cc < 0:
                            row[i] = f"${row[i]:.2f}*$"
                            n_neg_cc += 1
                        else:
                            row[i] = f"${row[i]:.2f}\\phantom{{*}}$"
                else: #fill with nan_fill if not enough valid data
                    row[f'cc_{data_source_lower}'] = nan_fill
                    row[f'pe_{data_source_lower}'] = nan_fill

            rows.append(row)

        df = pd.DataFrame(rows, columns=columns) #create dataframe
        logger.info(df)

        # Add row of mean values
        mean_row = {'site_id': 'Mean'}
        for col in df.columns:
            if col != 'site_id':
                mean = _mean_exclude_invalid(df[col], nan_fill=nan_fill)
                mean_row[col] = f"{mean:.2f}"
                if col.startswith('sigma'):
                    mean_row[col] = f"{mean:.1f}"

        df.loc[len(df)] = mean_row
        #rename columns with LaTeX
        df = df.rename(columns=column_names)
        # Format and save df as .md and .tex files
        fname = os.path.join(DATA_DIR, "_results", f"{data_type.lower()}_table")
        logger.info(f"Writing {data_type} prediction comparison tables to {fname}.{{md,tex}}")

        # Markdown
        # Apply nan_remove to each cell before writing to markdown
        df_md = df.map(_nan_remove)
        logger.info(f"Writing {fname+".md"}")
        df_md.to_markdown(fname + ".md", index=False, floatfmt=".2f")

        # LaTeX
        if n_neg_cc > 0:
            tex_note = f'{n_neg_cc}/{n_sites} sites found with r < 0.'
            logger.info(tex_note)

        # TODO: Pass nan_fill to _nan_remove
        formatters = {col: _nan_remove for col in df.columns}
        df_tex = fix_latex(df, data_type, formatters=formatters)
        with open(fname + ".tex", "w") as f:
            logger.info(f"Writing {fname+".tex"}")
            f.write(df_tex)
        fname = os.path.join(paper_dir, "figures", "_results", f"{data_type.lower()}_table")
        with open(fname + ".tex", "w") as f:
            logger.info(f"Writing {fname+".tex"}")
            f.write(df_tex)

    return {data_type: df}


def _mean_exclude_invalid(series, nan_fill=-99999):
    # Excludes nan_fill values from mean calculation
    value = pd.to_numeric(series.astype(str).str.replace(r'[$*]|\\phantom\{.*?\}', '', regex=True), errors='coerce')
    valid = value[value != nan_fill]
    return np.mean(valid) if len(valid) > 0 else ''


def _nan_remove(s, nan_fill=-99999): 
    # Removes nan_fill values
    #print(s)
    return '' if s == nan_fill else s


def _column_names(data_type):

    tmp = {
        'GIC': {
            'site_id': 'Site ID',
            'sigma_data': r'$\sigma$ [A]',
            'sigma_tva': r'$\sigma_\text{TVA}$',
            'sigma_gmu': r'$\sigma_\text{Ref}$',
            'cc_tva': r'$\text{r}^2_\text{TVA}$',
            'cc_gmu': r'$\text{r}^2_\text{Ref}$',
            'pe_tva': r'$\text{pe}_\text{TVA}$',
            'pe_gmu': r'$\text{pe}_\text{Ref}$'
        },
        'B': {
                'site_id':'Site ID',
                'sigma_data':r'$\sigma$ [nT]',
                'sigma_swmf':r'$\sigma_\text{SWMF}$',
                'sigma_mage':r'$\sigma_\text{MAGE}$',
                'sigma_openggcm':r'$\sigma_\text{GGCM}$',
                'cc_swmf':r'$\text{r}^2_\text{SWMF}$',
                'cc_mage':r'$\text{r}^2_\text{MAGE}$',
                'cc_openggcm':r'$\text{r}^2_\text{GGCM}$',
                'pe_swmf':r'$\text{pe}_\text{SWMF}$',
                'pe_mage':r'$\text{pe}_\text{MAGE}$',
                'pe_openggcm':r'$\text{pe}_\text{GGCM}$'
            }
        }

    return tmp[data_type]