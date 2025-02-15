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

def savefig(fname):
  if not os.path.exists(os.path.dirname(fname)):
    os.makedirs(os.path.dirname(fname))

  print(f"Writing {fname}.png")
  plt.savefig(f'{fname}.png', dpi=600, bbox_inches="tight")
"""
plt.scatter(df['dist(km)'], np.abs(df['cc']))
plt.xlabel('Distance (km)')
plt.ylabel('|cc|')
plt.grid(True)
savefig(os.path.join(results_dir, 'cc_vs_dist_scatter'))
plt.close()

avg_std = np.mean(df[['std_1', 'std_2']], axis=1)
plt.scatter(avg_std, np.abs(df['cc']))
plt.xlabel('Average standard deviation (A)')
plt.ylabel('|cc|')
plt.grid(True)
savefig(os.path.join(results_dir, 'cc_vs_std_scatter'))
plt.close()
"""
plt.scatter(np.abs(df['beta_diff']), np.abs(df['cc']))
plt.xlabel(r'|$\Delta \log_{10} (\beta)$|')
plt.ylabel('|cc|')
plt.grid(True)
savefig(os.path.join(results_dir, 'cc_vs_beta_scatter'))
plt.show()
plt.close()


####################################################################
# Site scatter

def site_plots(info_df, cc_df):

    def plot_cc(site_id, cc_df, distance=True):
        cc = []
        dist = []
        avg_std = []
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
        if distance == True:
            plt.scatter(dist, np.abs(cc))
            plt.xlabel('Distance (km)')
        else:
            plt.scatter(avg_std, np.abs(cc))
            plt.xlabel('Average standard deviation (A)')
        plt.ylabel('|cc|')
        plt.ylim(0, 1)
        plt.title(site_id)
        plt.grid(True)


    # Plotting maps and cc plots for each site
    for idx_1, row in info_df.iterrows():
        site_1_id = row['site_id']
        if site_1_id not in sites:
            continue

        # plotting cc vs distance
        plot_cc(site_1_id, cc_df)
        savefig(site_1_id, 'cc_vs_dist_scatter')
        plt.close()

        # plotting cc vs standard deviation
        plot_cc(site_1_id, cc_df, distance=False)
        savefig(site_1_id, 'cc_vs_std_scatter')
        plt.close()