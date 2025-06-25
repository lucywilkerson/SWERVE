import os
import json
import pickle

import numpy as np
import numpy.ma as ma
import pandas as pd

import utilrsw

from swerve import DATA_DIR, subset, plt_config, LOG_KWARGS, logger, format_df

logger = logger(**LOG_KWARGS)

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

    # Delete, e.g., {"nearest_sim_site": "86958445"} "data_source" entries.
    # They will eventually be removed from the info.json file.
    for sid in info_dict.keys():
        for data_type in info_dict[sid].keys():
            for data_class in info_dict[sid][data_type].keys():
                for idx, data_source in enumerate(info_dict[sid][data_type][data_class]):
                    if not isinstance(data_source, str):
                        del info_dict[sid][data_type][data_class][idx]

  print(f"Reading {all_file}")
  with open(all_file, 'rb') as f:
    data_all = pickle.load(f)

  return info_dict, data_all

def stats_measured(data_meas):
    return {
            'std': np.nanstd(data_meas),
            '|max|': np.abs(np.nanmax(data_meas)),
            'n': len(data_meas),
            'n_valid': np.sum(~np.isnan(data_meas))
    }

def stats_model(sid, data_meas, data_calc):
    valid = ~np.isnan(data_meas) & ~np.isnan(data_calc)
    stats_nan = {
            'std': np.nan,
            'cc': np.nan,
            'pe': np.nan,
            'n': len(data_calc),
            'n_valid': np.sum(valid)
    }
    if np.sum(valid) < 3:
        logger.warning("  Not enough valid data. Skipping.")
        return stats_nan

    if np.all(data_calc == 0):
        logger.warning("  All calculated data is zero. Skipping.")
        return stats_nan

    std_calc = np.nanstd(data_calc)
    cc = np.corrcoef(data_meas[valid], data_calc[valid])
    if cc[0,1] < 0:
        data_calc = -data_calc
    numer = np.sum((data_meas[valid] - data_calc[valid])**2)
    denom = np.sum((data_meas[valid] - data_meas[valid].mean())**2)
    pe = 1-numer/denom

    return {
            'std': std_calc,
            'cc': cc[0,1],
            'pe': pe,
            'n': len(data_calc),
            'n_valid': np.sum(valid)
    }

def stats2df(stats_dict):
    # Convert stats to DataFrame
    rows = []
    for sid, sid_stats in stats_dict.items():
        if 'GIC' not in sid_stats:
            logger.warning(f"  No GIC data for {sid}. Skipping.")
            continue

        data_type_stats = sid_stats['GIC']
        if 'measured' and 'calculated' not in data_type_stats:
            continue
        row = {'Site ID': sid}
        # Measured stats
        if 'measured' in data_type_stats:
            row[r'$\sigma$ [A]'] = data_type_stats['measured'].get('std', np.nan)
        # Calculated stats (add each data_source as separate columns)
        if 'calculated' in data_type_stats:
            for data_source, calc_stats in data_type_stats['calculated'].items():
                data_source = data_source.replace('GMU', 'Ref')
                row[fr'{data_source} $\sigma_\text{{{data_source}}}$ [A]'] = calc_stats.get('std', np.nan)
                row[fr'{data_source} $\text{{cc}}^2_\text{{{data_source}}}$'] = calc_stats.get('cc', np.nan)**2
                row[fr'{data_source} $\text{{pe}}_\text{{{data_source}}}'] = calc_stats.get('pe', np.nan)
        rows.append(row)

    gic_df = pd.DataFrame(rows)
    return gic_df

def extract_calc(data_calc, data_type, data_source, data_sources):
    #logger.info(f"  Computing stats for calculated ({data_source}) data")
    for idx in range(len(data_sources)):
        if data_source == data_sources[idx]:
            break
    time_calc = data_all[sid][data_type]['calculated'][idx]['original']['time']
    data_calc = data_all[sid][data_type]['calculated'][idx]['original']['data']

    if data_calc.ndim == 2 and data_type == 'GIC':
        # See https://github.com/lucywilkerson/2024-May-Storm-data/tree/main/dennies_gic_comparison/nerc
        # Columns for explanation of other data columns in GMU/calculated
        data_calc = data_calc[:, 0]
        data_calc = np.array(data_calc).flatten()

    return time_calc, data_calc

info_dict, data_all = read(all_file)

stats = {}
rows = []
for sid in info_dict.keys():
    stats[sid] = {}
    for data_type in ['B']:
        if data_type not in info_dict[sid].keys():
            continue
        stats[sid][data_type] = {}
        print(f"{sid}/{data_type}")
        data_classes = info_dict[sid][data_type].keys()
        if 'measured' not in data_all[sid][data_type]:
            logger.warning("  No measured data. Skipping.\n")
            continue
        if data_all[sid][data_type]['measured'][0] is None:
            logger.warning("  No measured data. Skipping.\n")
            continue
        time_meas = data_all[sid][data_type]['measured'][0]['modified']['time']
        data_meas = data_all[sid][data_type]['measured'][0]['modified']['data']
        time_meas, data_meas = subset(time_meas, data_meas, start, stop)
        stats[sid][data_type]['measured'] = stats_measured(data_meas)
        if ('measured' and 'calculated') not in data_classes:
            #logger.warning(f"  No calculated data for {data_type}. Skipping.")
            logger.info(utilrsw.format_dict(stats[sid][data_type], indent=2))
        else:
            data_sources = info_dict[sid][data_type]['calculated']
            stats[sid][data_type]['calculated'] = {}
            for data_source in data_sources:
                time_calc, data_calc = extract_calc(data_all[sid][data_type]['calculated'], data_type, data_source, data_sources)
                time_calc, data_calc = subset(time_calc, data_calc, start, time_meas[-1])
                import pdb; pdb.set_trace()
                if data_type == 'B':
                    data_calc = np.power(data_calc[:, 0:2], 2)
                    data_meas = np.power(data_meas[:, 0:2], 2)
                stats[sid][data_type]['calculated'][data_source] = stats_model(sid, data_meas, data_calc)

            logger.info(utilrsw.format_dict(stats[sid][data_type], indent=2))


gic_df = stats2df(stats)
logger.info(gic_df)

# Keep only rows for Bull Run and Montgomery
TVA_sites = ['Bull Run', 'Montgomery', 'Union', 'Widows Creek']
gic_df = gic_df[gic_df['Site ID'].isin(TVA_sites)]

# Format cells for markdown and LaTeX output
gic_df = format_df(gic_df)

fname = os.path.join(DATA_DIR, "_results", "gic_table")
print(f"Writing GIC prediction comparison tables to {fname}.{{md,tex}}")
gic_df.to_markdown(fname + ".md", index=False, floatfmt=".2f")
gic_df.to_latex(fname + ".tex", index=False, escape=False)
