import os
import pandas as pd
import pickle

import numpy as np
import matplotlib.pyplot as plt


data_dir = os.path.join('..', '2024-AGU-data')
out_dir = os.path.join('..', '2024-AGU-data', '_map')

#reading in info.extended.csv
fname = os.path.join('info', 'info.extended.csv')
print(f"Reading {fname}")
df = pd.read_csv(fname).set_index('site_id')
info_df = pd.read_csv(fname)

# Filter out sites with error message
info_df = info_df[~info_df['error'].str.contains('', na=False)]
# TODO: Print number of GIC sites removed due to error and how many kept.
# Remove rows that don't have data_type = GIC and data_class = measured
info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
info_df.reset_index(drop=True, inplace=True)

# Read in cc data
pkl_file = os.path.join(data_dir, '_results', 'cc.pkl')
with open(pkl_file, 'rb') as file:
  print(f"Reading {pkl_file}")
  cc_rows = pickle.load(file)
cc_df = pd.DataFrame(cc_rows)
cc_df.reset_index(drop=True, inplace=True)


def site_filt(info_df, cc_df, cc_lim):
    print(f'Sites with no pairing |cc| > {cc_lim}:')
    def is_site_bad(site_id, cc_df):
        n_site = 0
        for idx, row in cc_df.iterrows():
            if row['site_1'] == site_id:
                site_2_id = row['site_2']
            elif row['site_2'] == site_id:
                site_2_id = row['site_1']
            else:
                continue
            cc = np.abs(row['cc'])
            if cc > cc_lim:
                break
            n_site += 1
        if n_site == 55:
            print(site_id)
    for idx_1, row in info_df.iterrows():
        site_1_id = row['site_id']
        is_site_bad(site_1_id, cc_df)

cc_max = 0.3
site_filt(info_df, cc_df, cc_max)

#################################################################################################
# defining avg |cc| for each site
for idx_1, row in info_df.iterrows():
    site_1_id = row['site_id']
    site_cc = []
    for idx_2, row in cc_df.iterrows():
        if row['site_1'] == site_1_id:
            site_2_id = row['site_2']
        elif row['site_2'] == site_1_id:
            site_2_id = row['site_1']
        else:
            continue
        cc = np.abs(row['cc'])
        site_cc.append(cc)
    avg_cc = np.mean(site_cc)
    info_df.loc[info_df['site_id'] == site_1_id, 'avg_cc'] = avg_cc #adding mean cc to info_df

# Adding min(avg_cc) to cc_df
for idx, row in cc_df.iterrows():
    site_1_id = row['site_1']
    site_2_id = row['site_2']
    avg_cc_1 = info_df.loc[info_df['site_id'] == site_1_id, 'avg_cc'].values[0]
    avg_cc_2 = info_df.loc[info_df['site_id'] == site_2_id, 'avg_cc'].values[0]
    cc_df.loc[idx, 'min_avg_cc'] = min(avg_cc_1, avg_cc_2)

# Making scatter plot! (code from plot_scatter.py)
results_dir = os.path.join('..', '2024-AGU-data', '_results')
fmts = ['png', 'pdf']
def savefig(fdir, fname, fmts=fmts):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = os.path.join(fdir, fname)

    for fmt in fmts:
        print(f"    Saving {fname}.{fmt}")
        if fmt == 'png':
            plt.savefig(f'{fname}.{fmt}', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def scatter_with_colorbar(df, color_col, cbar_label, plot_title, file_name):
    # Define 10 discrete color bins for the color column
    bins = np.linspace(np.abs(df[color_col]).min(), np.abs(df[color_col]).max(), 10)
    norm = plt.Normalize(bins.min(), bins.max())
    cmap = plt.cm.get_cmap('viridis', len(bins) - 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Plotting scatter with colorbar
    fig, ax = plt.subplots(figsize=(12, 5))
    sc = ax.scatter(df['dist(km)'], np.abs(df['cc']), c=np.abs(df[color_col]), cmap=cmap, norm=norm)
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('|cc|')
    #ax.set_title(plot_title)
    ax.grid(True)
    
    # Set up colorbar
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Position for the colorbar
    cbar = plt.colorbar(sm, cax=cax, ticks=bins, label=cbar_label)
    
    # Add dots to the colorbar
    #cbar.ax.clear()
    #for i, b in enumerate(bins[:-1]):
        #cax.plot([0.5], [b], 'o', color=cmap(i), markersize=5, transform=cax.get_yaxis_transform(), clip_on=False)
    cbar.set_ticks(bins)
    cbar.set_ticklabels([f'{b:.2f}' for b in bins])
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_label(cbar_label)
    cbar.ax.xaxis.set_visible(False)
    savefig(results_dir, file_name)
    plt.close()

scatter_with_colorbar(cc_df, 'min_avg_cc', r'min mean |cc|', 'CC vs Distance with Min |cc| Colorbar', 'cc_vs_dist_vs_min_scatter')