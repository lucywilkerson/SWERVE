import os
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

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

"""# Scatter plots for all sites

plt.scatter(df['dist(km)'], np.abs(df['cc']))
plt.xlabel('Distance [km]')
plt.ylabel('|cc|')
plt.grid(True)
savefig(results_dir, 'cc_vs_dist_scatter')
plt.close()

avg_std = np.mean(df[['std_1', 'std_2']], axis=1)
plt.scatter(avg_std, np.abs(df['cc']))
plt.xlabel('Average standard deviation [A]')
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

plt.scatter(np.abs(df['volt_diff(kV)']), np.abs(df['cc']))
nan_volt_diff = df['volt_diff(kV)'].isna().sum()
plt.text(0.10, 0.95, f"NaN values: {nan_volt_diff}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.5))
plt.xlabel(r'|$\Delta$V| [kV]')
plt.ylabel('|cc|')
plt.grid(True)
savefig(results_dir, 'cc_vs_volt_scatter')
plt.close()


# Scatter plots with colorbars!

# Define 10 discrete color bins for beta
bins = np.linspace(np.abs(df['beta_diff']).min(), np.abs(df['beta_diff']).max(), 10)
norm = plt.Normalize(bins.min(), bins.max())
cmap = plt.cm.get_cmap('viridis', len(bins) - 1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
colors = np.digitize(np.abs(df['beta_diff']), bins)

# Plotting cc vs dist w beta colorbar
fig, ax = plt.subplots(figsize=(12, 5))
sc = ax.scatter(df['dist(km)'], np.abs(df['cc']), c=colors, cmap=cmap, norm=norm)
ax.set_xlabel('Distance (km)')
ax.set_ylabel('|cc|')
ax.grid(True)
# Set up colorbar
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Position for the colorbar
cbar = plt.colorbar(sm, cax=cax, ticks=bins, label=r'|$\Delta \log_{10} (\beta)$|')
# Add dots to the colorbar
cbar.ax.clear()
for i, b in enumerate(bins[:-1]):
    cax.plot([0.5], [b], 'o', color=cmap(i), markersize=5, transform=cax.get_yaxis_transform(), clip_on=False)
cbar.set_ticks(bins)
cbar.set_ticklabels([f'{b:.2f}' for b in bins])
cbar.ax.yaxis.set_label_position('right')
cbar.set_label(r'|$\Delta \log_{10} (\beta)$|')
cbar.ax.xaxis.set_visible(False)  
savefig(results_dir, 'cc_vs_dist_vs_beta_scatter')
plt.close()"""


###################################################################
# Scatters with grid coding :P

regions = df['region'].unique()
colors = {'East': 'blue', 'West': 'green', 'Central': 'red', 'ERCOT': 'purple', 'different': 'gray'}

pools = df['power_pool'].unique()
shapes = {pool: shape for pool, shape in zip(pools, ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h'])}

# Scatter plot for distance vs |cc| with region colors and pool shapes
fig, ax = plt.subplots()
for region in regions:
    subset = df[df['region'] == region]
    for pool in pools:
        pool_subset = subset[subset['power_pool'] == pool]
        ax.scatter(pool_subset['dist(km)'], np.abs(pool_subset['cc']), label=f"{region} - {pool}", color=colors.get(region), marker=shapes.get(pool))
plt.xlabel('Distance [km]')
plt.ylabel('|cc|')
# Create separate legends
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
region_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[region], markersize=10) for region in regions]
pool_handles = [plt.Line2D([0], [0], marker=shapes[pool], color='w', markerfacecolor='k', markersize=10) for pool in pools]
region_labels = regions
pool_labels = pools
first_legend = plt.legend(region_handles, region_labels, title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.add_artist(first_legend)
plt.legend(pool_handles, pool_labels, title='Power Pool', bbox_to_anchor=(1.05, 0.3), loc='center left')
plt.grid(True)
savefig(results_dir, 'cc_vs_dist_grid_scatter')
plt.close()

# Scatter plot for average standard deviation vs |cc| with region colors and pool shapes
fig, ax = plt.subplots()
avg_std = np.mean(df[['std_1', 'std_2']], axis=1)
for region in regions:
    subset = df[df['region'] == region]
    for pool in pools:
        pool_subset = subset[subset['power_pool'] == pool]
        ax.scatter(np.mean(pool_subset[['std_1', 'std_2']], axis=1), np.abs(pool_subset['cc']), label=f"{region} - {pool}", color=colors.get(region), marker=shapes.get(pool))
plt.xlabel('Average standard deviation [A]')
plt.ylabel('|cc|')
# Create separate legends
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
region_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[region], markersize=10) for region in regions]
pool_handles = [plt.Line2D([0], [0], marker=shapes[pool], color='w', markerfacecolor='k', markersize=10) for pool in pools]
region_labels = regions
pool_labels = pools
first_legend = plt.legend(region_handles, region_labels, title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.add_artist(first_legend)
plt.legend(pool_handles, pool_labels, title='Power Pool', bbox_to_anchor=(1.05, 0.3), loc='center left')
plt.grid(True)
savefig(results_dir, 'cc_vs_std_grid_scatter')
plt.close()

# Scatter plot for |beta_diff| vs |cc| with region colors and pool shapes
fig, ax = plt.subplots()
for region in regions:
    subset = df[df['region'] == region]
    for pool in pools:
        pool_subset = subset[subset['power_pool'] == pool]
        ax.scatter(np.abs(pool_subset['beta_diff']), np.abs(pool_subset['cc']), label=f"{region} - {pool}", color=colors.get(region), marker=shapes.get(pool))
plt.xlabel(r'|$\Delta \log_{10} (\beta)$|')
plt.ylabel('|cc|')
# Create separate legends
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
region_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[region], markersize=10) for region in regions]
pool_handles = [plt.Line2D([0], [0], marker=shapes[pool], color='w', markerfacecolor='k', markersize=10) for pool in pools]
region_labels = regions
pool_labels = pools
first_legend = plt.legend(region_handles, region_labels, title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.add_artist(first_legend)
plt.legend(pool_handles, pool_labels, title='Power Pool', bbox_to_anchor=(1.05, 0.3), loc='center left')
plt.grid(True)
savefig(results_dir, 'cc_vs_beta_grid_scatter')
plt.close()

# Scatter plot for |volt_diff(kV)| vs |cc| with region colors and pool shapes
fig, ax = plt.subplots()
for region in regions:
    subset = df[df['region'] == region]
    for pool in pools:
        pool_subset = subset[subset['power_pool'] == pool]
        ax.scatter(np.abs(pool_subset['volt_diff(kV)']), np.abs(pool_subset['cc']), label=f"{region} - {pool}", color=colors.get(region), marker=shapes.get(pool))
nan_volt_diff = df['volt_diff(kV)'].isna().sum()
plt.text(0.10, 0.95, f"NaN values: {nan_volt_diff}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.5))
plt.xlabel(r'|$\Delta$V| [kV]')
plt.ylabel('|cc|')
# Create separate legends
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
region_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[region], markersize=10) for region in regions]
pool_handles = [plt.Line2D([0], [0], marker=shapes[pool], color='w', markerfacecolor='k', markersize=10) for pool in pools]
region_labels = regions
pool_labels = pools
first_legend = plt.legend(region_handles, region_labels, title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.add_artist(first_legend)
plt.legend(pool_handles, pool_labels, title='Power Pool', bbox_to_anchor=(1.05, 0.3), loc='center left')
plt.grid(True)
savefig(results_dir, 'cc_vs_volt_grid_scatter')
plt.close()

exit()

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