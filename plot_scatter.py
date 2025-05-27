import os
import pickle
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600

data_dir = os.path.join('..', '2024-May-Storm-data')
results_dir = os.path.join('..', '2024-May-Storm-data', '_results')
paper_dir = os.path.join('..','2024-May-Storm-paper')

pkl_file = os.path.join(results_dir, 'cc.pkl')
print(f"Reading {pkl_file}")
with open(pkl_file, 'rb') as file:
  df = pickle.load(file)

fmts = ['png','pdf']
def savefig(fdir, fname, fmts=fmts):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = os.path.join(fdir, fname)

    for fmt in fmts:
        print(f"    Saving {fname}.{fmt}")
        plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def savefig_paper(fname, sub_dir="", fmts=['png','pdf']):
  fdir = os.path.join(paper_dir, sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    print(f"    Saving {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def add_subplot_label(ax, label, loc=(-0.15, 1)):
  ax.text(*loc, label, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

Poster = False # set to be true to generate poster figs
paper = True # set true to generate paper figs
colorbar_scatter = True # set true to generate colorbar plots
grid_scatter = False # set true to generate grid plots
site_scatter = False # set true to generate scatter plots for each site

# Scatter plots for all sites

def plot_avg_line(x, y, bins=23, color='k', marker='o', label='Average in bins', **kwargs):
    # Plots a line of average y values over binned x values

    # Bin data and calculate averages
    points_per_bin = int(len(x)/bins)
    sorted_x, sorted_y = zip(*sorted(zip(x, y)))
    sorted_x = np.array(sorted_x)
    sorted_y = np.array(sorted_y)

    bin_edges = [sorted_x[i * points_per_bin] for i in range(bins)] + [sorted_x[-1]]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]
    avg_y = [np.mean(sorted_y[(sorted_x >= bin_edges[i]) & (sorted_x < bin_edges[i + 1])]) for i in range(bins)]

    # Plot average line
    plt.plot(bin_centers, avg_y, color=color, marker=marker, label=label, **kwargs)


# Scatter plot
scatter_kwargs = {'color': 'gray', 'alpha': 0.9}

plt.scatter(df['dist(km)'], np.abs(df['cc']), **scatter_kwargs)
plot_avg_line(df['dist(km)'], np.abs(df['cc']))
plt.xlabel('Distance [km]')
plt.ylabel('|cc|')
plt.grid(True)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
plt.gca().xaxis.grid(True, linestyle='--', which='minor')
plt.legend(loc='upper right')
savefig(results_dir, 'cc_vs_dist_scatter')
if paper:
    add_subplot_label(plt.gca(), 'a)')
    savefig_paper('cc_vs_dist_scatter', 'scatter')
plt.close()

avg_std = np.mean(df[['std_1', 'std_2']], axis=1)
plt.scatter(avg_std, np.abs(df['cc']), **scatter_kwargs)
plot_avg_line(avg_std, np.abs(df['cc']))
plt.xlabel('Average standard deviation [A]')
plt.ylabel('|cc|')
plt.grid(True)
plt.legend(loc='upper right')
savefig(results_dir, 'cc_vs_std_scatter')
if paper:
    savefig_paper('cc_vs_std_scatter', 'scatter')
plt.close()

plt.scatter(np.abs(df['log_beta_diff']), np.abs(df['cc']), **scatter_kwargs)
plot_avg_line(np.abs(df['log_beta_diff']), np.abs(df['cc']))
plt.xlabel(r'|$\Delta \log_{10} (\beta)$|')
plt.ylabel('|cc|')
plt.grid(True)
plt.legend(loc='upper right')
savefig(results_dir, 'cc_vs_beta_scatter')
if paper:
    add_subplot_label(plt.gca(), 'c)')
    savefig_paper('cc_vs_beta_scatter', 'scatter')
plt.close()

plt.scatter(np.abs(df['volt_diff(kV)']), np.abs(df['cc']), **scatter_kwargs)
nan_volt_diff = df['volt_diff(kV)'].isna().sum()
#plt.text(0.10, 0.95, f"NaN values: {nan_volt_diff}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.5))
plt.xlabel(r'|$\Delta$V| [kV]')
plt.ylabel('|cc|')
plt.grid(True)
#plt.legend(loc='upper right')
savefig(results_dir, 'cc_vs_volt_scatter')
if paper:
    add_subplot_label(plt.gca(), 'd)')
    savefig_paper('cc_vs_volt_scatter', 'scatter')
plt.close()

plt.scatter(np.abs(df['lat_diff']), np.abs(df['cc']), **scatter_kwargs)
plot_avg_line(np.abs(df['lat_diff']), np.abs(df['cc']))
plt.xlabel(r'$\Delta$ Latitude [deg]')
plt.ylabel('|cc|')
plt.grid(True)
plt.legend(loc='upper right')
savefig(results_dir, 'cc_vs_lat_scatter')
if paper:
    add_subplot_label(plt.gca(), 'b)')
    savefig_paper('cc_vs_lat_scatter', 'scatter')
plt.close()

# scatter plots not in paper
if not paper:
    plt.scatter(np.abs(df['log_beta_diff']), avg_std)
    plt.xlabel(r'|$\Delta \log_{10} (\beta)$|')
    plt.ylabel('Average standard deviation [A]')
    plt.grid(True)
    savefig(results_dir, 'std_vs_beta_scatter')
    plt.close()

    plt.scatter(np.abs(df['lat_diff']), avg_std)
    plt.xlabel(r'$\Delta$ Latitude [deg]')
    plt.ylabel('Average standard deviation [A]')
    plt.grid(True)
    savefig(results_dir, 'std_vs_lat_scatter')
    plt.close()


# for poster
if Poster: 
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.scatter(df['dist(km)'], np.abs(df['cc']))
    plt.xlabel('Distance [km]')
    plt.ylabel('|cc|')
    plt.grid(True)
    # outlining points
    for idx, row in df.iterrows():
        if row['site_1'] == '10181' and row['site_2'] == '10099':
            plt.scatter(row['dist(km)'], np.abs(row['cc']), facecolor='none', edgecolor='k')
        elif row['site_1'] == 'Widows Creek' and row['site_2'] == 'Bradley':
            plt.scatter(row['dist(km)'], np.abs(row['cc']), facecolor='none', edgecolor='k')
    # empty colorbar
    cmap = mpl.cm.viridis  
    norm = mpl.colors.Normalize(vmin=0, vmax=1)  
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.clear()
    cbar.ax.tick_params(size=0)
    cbar.ax.set_xticks([])
    cbar.ax.set_yticks([])
    cbar.outline.set_visible(False)
    savefig(results_dir, 'cc_vs_dist_scatter_poster')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.scatter(np.abs(df['log_beta_diff']), np.abs(df['cc']))
    plt.xlabel(r'|$\Delta \log_{10} (\beta)$|')
    plt.ylabel('|cc|')
    plt.grid(True)
    # empty colorbar
    cmap = mpl.cm.viridis  
    norm = mpl.colors.Normalize(vmin=0, vmax=1)  
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.clear()
    cbar.ax.tick_params(size=0)
    cbar.ax.set_xticks([])
    cbar.ax.set_yticks([])
    cbar.outline.set_visible(False)
    savefig(results_dir, 'cc_vs_beta_scatter_poster')
    plt.close()

#########################################################################################
# Scatter plots with colorbars!

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
    savefig_paper(file_name,'scatter')
    plt.close()

# Generating plots
if colorbar_scatter:
    scatter_with_colorbar(df, 'log_beta_diff', r'|$\Delta \log_{10} (\beta)$|', 'CC vs Distance with Beta Colorbar', 'cc_vs_dist_vs_beta_scatter')
    scatter_with_colorbar(df, 'volt_diff(kV)', r'|$\Delta V$| [kV]', 'CC vs Distance with Line Voltage Colorbar', 'cc_vs_dist_vs_volt_scatter')
    scatter_with_colorbar(df, 'lat_diff', r'$\Delta$ Latitude [deg]', 'CC vs Distance with Latitude Colorbar', 'cc_vs_dist_vs_lat_scatter')
    scatter_with_colorbar(df, 'min_avg_cc', r'min mean |cc|', 'CC vs Distance with Min |cc| Colorbar', 'cc_vs_dist_vs_min_scatter')


###################################################################
# Scatters with grid coding :P

regions = df['region_1'].unique()
colors = {'East': 'blue', 'West': 'green', 'Central': 'red', 'ERCOT': 'purple'}

pools = df['power_pool_2'].unique()
shapes = {pool: shape for pool, shape in zip(pools, ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X'])}
    
def plot_grid_scatter(x, y, x_label, y_label, region_names=['East', 'West', 'Central', 'ERCOT']):
    #makes 4-panel plot w region colors and pool shapes

    #define function to filter region
    def filter_region(region_name, row):
        if row['region_1'] != region_name and row['region_2'] != region_name:
            return None
        elif row['region_1'] == region_name and row['region_2'] != region_name:
            return row['region_2']
        elif row['region_1'] != region_name and row['region_2'] == region_name:
            return row['region_1']
        else:
            return region_name

    # define function to make legend
    def add_grid_legends(fig, regions, pools, colors, shapes):
        region_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[region], markersize=10) for region in regions]
        pool_handles = [plt.Line2D([0], [0], marker=shapes[pool], color='w', markerfacecolor='k', markersize=10) for pool in pools]
        region_labels = regions
        pool_labels = pools
        fig.legend(region_handles, region_labels, title='2nd Region', bbox_to_anchor=(1.03, 0.7), loc='upper left')
        fig.legend(pool_handles, pool_labels, title='1st Power Pool', bbox_to_anchor=(1.01, 0.3), loc='center left')

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for ax, region in zip(axs, region_names):
        for idx, row in df.iterrows():
            reg_b = filter_region(region, row)
            if reg_b is None:
                continue
            pool = row['power_pool_1'] if row['region_1'] == region else row['power_pool_2']
            ax.scatter(x[idx], y[idx], label=f"{region} - {pool}", color=colors.get(reg_b), marker=shapes.get(pool))
        num_points = sum(df.apply(lambda row: filter_region(region, row) is not None, axis=1))
        ax.set_title(f"1st Region {region} ({num_points} site pairs)")
        if ax in axs[-2:]:
            ax.set_xlabel(x_label)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(y_label)
        ax.grid(True)

    # Set the same limits for all axes
    for ax in axs:
        xlims = [ax.get_xlim() for ax in axs]
        ylims = [ax.get_ylim() for ax in axs]
        x_min = min(lim[0] for lim in xlims)
        x_max = max(lim[1] for lim in xlims)
        y_min = min(lim[0] for lim in ylims)
        y_max = max(lim[1] for lim in ylims)
        for ax in axs:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    add_grid_legends(fig, regions, pools, colors, shapes)

    plt.tight_layout()

if grid_scatter:
    # Four panel plot for distance vs |cc| with region colors and pool shapes
    plot_grid_scatter(df['dist(km)'], np.abs(df['cc']), 'Distance [km]', '|cc|')
    savefig(results_dir, 'cc_vs_dist_grid_scatter')
    plt.close()

    # Four panel plot for average standard deviation vs |cc| with region colors and pool shapes
    avg_std = np.zeros(len(df))
    for idx, row in df.iterrows():
        avg_std[idx] = np.mean([row['std_1'], row['std_2']])
    plot_grid_scatter(avg_std, np.abs(df['cc']), 'Average standard deviation [A]', '|cc|')
    savefig(results_dir, 'cc_vs_std_grid_scatter')
    plt.close()

    # Four panel plot for |log_beta_diff| vs |cc| with region colors and pool shapes
    plot_grid_scatter(np.abs(df['log_beta_diff']), np.abs(df['cc']), r'|$\Delta \log_{10} (\beta)$|', '|cc|')
    savefig(results_dir, 'cc_vs_beta_grid_scatter')
    plt.close()

    # Four panel plot for |volt_diff(kV)| vs |cc| with region colors and pool shapes
    plot_grid_scatter(np.abs(df['volt_diff(kV)']), np.abs(df['cc']), r'|$\Delta$V| [kV]', '|cc|')
    savefig(results_dir, 'cc_vs_volt_grid_scatter')
    plt.close()

    # Four panel plot for |lat_diff| vs |cc| with region colors and pool shapes
    plot_grid_scatter(np.abs(df['lat_diff']), np.abs(df['cc']), r'$\Delta$ Latitude [deg]', '|cc|')
    savefig(results_dir, 'cc_vs_lat_grid_scatter')
    plt.close()


####################################################################
# Site scatter
if site_scatter:
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
                beta.append(row['log_beta_diff'])
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