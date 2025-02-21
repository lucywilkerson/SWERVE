import os
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data_dir = os.path.join('..', '2024-AGU-data')
results_dir = os.path.join('..', '2024-AGU-data', '_results')

pkl_file = os.path.join(results_dir, 'cc.pkl')
print(f"Reading {pkl_file}")
with open(pkl_file, 'rb') as file:
  df = pickle.load(file)

def savefig(fdir, fname, fmts=['png']):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = os.path.join(fdir, fname)

    for fmt in fmts:
        print(f"    Saving {fname}.{fmt}")
        if fmt == 'png':
            plt.savefig(f'{fname}.{fmt}', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

# Scatter plots for all sites

plt.scatter(df['dist(km)'], np.abs(df['cc']))
plt.xlabel('Distance (km)')
plt.ylabel('|cc|')
plt.grid(True)
savefig(results_dir, 'cc_vs_dist_scatter')
plt.close()

avg_std = np.mean(df[['std_1', 'std_2']], axis=1)
plt.scatter(avg_std, np.abs(df['cc']))
plt.xlabel('Average standard deviation (A)')
plt.ylabel('|cc|')
plt.grid(True)
savefig(results_dir, 'cc_vs_std_scatter')
plt.close()

plt.scatter(np.abs(df['beta_diff']), np.abs(df['cc']))
plt.xlabel(r'|$\Delta \log_{10} (\beta)$|')
plt.ylabel('|cc|')
plt.grid(True)
savefig(results_dir, 'cc_vs_beta_scatter')
plt.close()

####################################################################
# Site scatter

def site_plots(info_df, cc_df, sites):

    def plot_cc(site_id, cc_df, type='dist'):
        cc = []
        dist = []
        avg_std = []
        beta = []
        for idx, row in cc_df.iterrows():
            if row['site_1'] == site_id:
                site_2_id = row['site_2']
            elif row['site_2'] == site_id:
                site_2_id = row['site_1']
            else:
                continue
            cc.append(row['cc'])
            dist.append(row['dist(km)'])
            avg_std.append(np.mean([row['std_1'], row['std_2']]))
            beta.append(row['beta_diff'])
        if type == 'dist':
            plt.scatter(dist, np.abs(cc))
            plt.xlabel('Distance [km]')
        elif type == 'std':
            plt.scatter(avg_std, np.abs(cc))
            plt.xlabel('Average standard deviation [A]')
        elif type == 'beta':
            plt.scatter(np.abs(beta), np.abs(cc))
            plt.xlabel(r'|$\Delta \log_{10} (\beta)$|')
        plt.ylabel('|cc|')
        plt.ylim(0, 1)
        plt.title(site_id)
        plt.grid(True)

    # Plotting maps and cc plots for each site
    for idx_1, row in info_df.iterrows():
        site_1_id = row['site_id']
        if site_1_id not in sites:
            continue

        # set up directory to save
        sid = site_1_id
        sub_dir=""
        fdir = os.path.join(data_dir, '_processed', sid.lower().replace(' ', ''), sub_dir)

        # plotting cc vs distance
        plot_cc(site_1_id, cc_df, type='dist')
        savefig(fdir, 'cc_vs_dist_scatter')
        plt.close()

        # plotting cc vs standard deviation
        plot_cc(site_1_id, cc_df, type='std')
        savefig(fdir, 'cc_vs_std_scatter')
        plt.close()

        # plotting cc vs standard deviation
        plot_cc(site_1_id, cc_df, type='beta')
        savefig(fdir, 'cc_vs_beta_scatter')
        plt.close()

#reading in info.csv
fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)
# Remove rows that have errors
info_df = info_df[~info_df['error'].str.contains('', na=False)]
# Remove rows that don't have data_type = GIC and data_class = measured
info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
# List "good" GIC sites
sites = info_df['site_id'].tolist()


site_plots(info_df, df, sites)