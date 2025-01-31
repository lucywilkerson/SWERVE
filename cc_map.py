import os
import pandas as pd
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

data_dir = os.path.join('..', '2024-AGU-data')

state = True # Show political boundaries

transform = ccrs.PlateCarree()
projection = ccrs.Miller()
crs = ccrs.PlateCarree()

def savefig(sid, fname, sub_dir="", fmts=['png']):
  fdir = os.path.join(data_dir, '_processed', sid.lower().replace(' ', ''), sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    print(f"    Saving {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def add_features(ax, state):
    # Add coastlines and other features
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    lakes = cfeature.NaturalEarthFeature('physical', 'lakes', '50m',
                                         edgecolor='face',
                                         facecolor='lightblue')
    ax.add_feature(lakes)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    if state == True:
        ax.add_feature(cfeature.STATES, linewidth=0.5)

def cc_vs_dist_map(cc_df):

    def add_pair_lines(ax, site_1_id, site_2_id, color='k'):
        # function to draw lines between pairs
        # TODO: merge with map_cc
        site_1_lat = info_df.loc[info_df['site_id'] == site_1_id, 'geo_lat'].values[0]
        site_1_lon = info_df.loc[info_df['site_id'] == site_1_id, 'geo_lon'].values[0]

        site_2_lat = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lat'].values[0]
        site_2_lon = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lon'].values[0]

        ax.plot([site_1_lon, site_2_lon], [site_1_lat, site_2_lat],
                color=color, transform=transform)

    # mapping top and bottom site pairs
    cc_lims = [0.8,0.6,0.4,0.2,0.01]
    # distance limit for cc < 0.4
    dist_lim = 1000

    #setting up maps
    fig, axs = plt.subplots(2,3, figsize=(13, 5), subplot_kw={'projection': projection})
    for ax in axs.flat:
        add_features(ax, state)
        ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)

    #values to count number of pairs
    num_pairs = np.zeros(6)
    # looping over pairs
    for idx, row in cc_df.iterrows():
        cc = np.abs(row['cc'])
        site_1_id = row['site_1']
        site_2_id = row['site_2']
        if cc > cc_lims[0]:
            add_pair_lines(axs[0,0], site_1_id, site_2_id, color='r')
            num_pairs[0] += 1
        elif cc > cc_lims[1]:
            add_pair_lines(axs[0,1], site_1_id, site_2_id)
            num_pairs[1] += 1
        elif cc > cc_lims[2]:
            add_pair_lines(axs[0,2], site_1_id, site_2_id)
            num_pairs[2] += 1
        elif cc > cc_lims[3]:
            if row['dist(km)'] > dist_lim:
                continue
            add_pair_lines(axs[1,0], site_1_id, site_2_id)
            num_pairs[3] += 1
        elif cc > cc_lims[4]:
            if row['dist(km)'] > dist_lim:
                continue
            add_pair_lines(axs[1,1], site_1_id, site_2_id)
            num_pairs[4] += 1
        elif cc < cc_lims[4]:
            if row['dist(km)'] > dist_lim:
                continue
            add_pair_lines(axs[1,2], site_1_id, site_2_id)
            num_pairs[5] += 1

    labels = [
        f"{int(num_pairs[0])} sites with |cc| > {cc_lims[0]}",
        f"{int(num_pairs[1])} sites with {cc_lims[0]} > |cc| > {cc_lims[1]}",
        f"{int(num_pairs[2])} sites with {cc_lims[1]} > |cc| > {cc_lims[2]}",
        f"{int(num_pairs[3])} sites with {cc_lims[2]} > |cc| > {cc_lims[3]}",
        f"{int(num_pairs[4])} sites with {cc_lims[3]} > |cc| > {cc_lims[4]}",
        f"{int(num_pairs[5])} sites with |cc| < {cc_lims[4]}"
    ]

    for ax, label in zip(axs.flat, labels):
        ax.text(0.5, -0.1, label, ha="center", transform=ax.transAxes, fontsize=8)
    plt.tight_layout()
    #plt.show()
    output_fname = os.path.join(data_dir, '_results', 'cc_vs_dist_map.png')
    if not os.path.exists(os.path.dirname(output_fname)):
        os.makedirs(os.path.dirname(output_fname))
    print(f"Saving {output_fname}")
    plt.savefig(output_fname, bbox_inches='tight')
    plt.close()

def std_map(info_df, cc_df):
    # plotting standard deviation
    # Create a figure and axes with a specific projection
    fig, ax = plt.subplots(1, figsize=(10, 8), subplot_kw={'projection': projection})
    # setting up map
    add_features(ax, state)
    # Set the extent of the map (USA)
    ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)
    plt.title('Standard deviation of GIC data')
    # plotting map w lines
    for idx_1, row_1 in info_df.iterrows():
        site_id = row_1['site_id']
        if site_id not in sites:
            continue
        site_lat = info_df.loc[info_df['site_id'] == site_id, 'geo_lat'].values[0]
        site_lon = info_df.loc[info_df['site_id'] == site_id, 'geo_lon'].values[0]
        for idx_2, row_2 in cc_df.iterrows():
            if row_2['site_1'] == site_id:
                std = row_2['std_1']
            else:
                continue
            ax.plot(site_lon, site_lat, color='k', marker='o', markersize=std, transform=transform)
    #plt.show()
    output_fname = os.path.join(data_dir, '_results', 'std_map.png')
    if not os.path.exists(os.path.dirname(output_fname)):
        os.makedirs(os.path.dirname(output_fname))
    print(f"Saving {output_fname}")
    plt.savefig(output_fname, bbox_inches='tight')
    plt.close()

def site_plots(info_df, cc_df):

    def map_cc(ax, site_id, cc_df, colors=False, lines=False):

        def add_cc_colors(cc):
            if cc < 0.2:
                color = 'red'
                mark = 2
            elif cc < 0.4:
                color = 'orange'
                mark = 2.75
            elif cc < 0.6:
                color = 'yellow'
                mark = 3.5
            elif cc < 0.8:
                color = 'lightgreen'
                mark = 4.25
            else:
                color = 'green'
                mark = 5
            return color, mark

        def add_cc_width(cc):

            if cc < 0.2:
                width = 1
            elif cc < 0.4:
                width = 2
            elif cc < 0.6:
                width = 3
            elif cc < 0.8:
                width = 4
            else:
                width = 5
            return width

        site_1_lat = info_df.loc[info_df['site_id'] == site_id, 'geo_lat'].values[0]
        site_1_lon = info_df.loc[info_df['site_id'] == site_id, 'geo_lon'].values[0]

        for idx, row in cc_df.iterrows():
            if row['site_1'] == site_id:
                site_2_id = row['site_2']
            elif row['site_2'] == site_id:
                site_2_id = row['site_1']
            else:
                continue
            cc = np.abs(row['cc'])
            site_2_lat = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lat'].values[0]
            site_2_lon = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lon'].values[0]
            if colors == True:
                col, marksiz= add_cc_colors(cc)
                ax.plot(site_2_lon, site_2_lat, color=col, 
                        marker='o', markersize=marksiz, transform=transform)
                prim_color='k'
            elif lines == True:
                width = add_cc_width(cc)
                ax.plot([site_1_lon, site_2_lon], [site_1_lat, site_2_lat],
                        color='k', linewidth=width, transform=transform)
                prim_color='r'
            else:
                ax.plot(site_2_lon, site_2_lat, 'bo', transform=transform)
                prim_color='k'
            ax.plot(site_1_lon, site_1_lat, color=prim_color, marker='*', transform=transform)

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

        # Create a figure and axes with a specific projection
        _, axs = plt.subplots(2, figsize=(10, 8), subplot_kw={'projection': projection})
        # setting up map
        for ax in axs.flat:
            add_features(ax, state)
            # Set the extent of the map (USA)
            ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)

        # plotting map w lines
        map_cc(axs[0], site_1_id, cc_df, lines=True)
        # plotting map w colors
        map_cc(axs[1], site_1_id, cc_df, colors=True)
        plt.suptitle(site_1_id)
        #plt.show()
        savefig(site_1_id, 'cc_vs_dist_map')
        plt.close()

        # plotting cc vs distance
        plot_cc(site_1_id, cc_df)
        #plt.show()
        savefig(site_1_id, 'cc_vs_dist_scatter')
        plt.close()

        # plotting cc vs standard deviation
        plot_cc(site_1_id, cc_df, distance=False)
        #plt.show()
        savefig(site_1_id, 'cc_vs_std_scatter')
        plt.close()

## Function to add transmission lines to the map ##
# TODO: incorporate into other plotting code as switch
def add_trans_lines(ax):
    # US Transmission lines
    data_path = os.path.join('..', '2024-AGU-data', 'Electric__Power_Transmission_Lines')
    data_name = 'Electric__Power_Transmission_Lines.shp'
    print(f"Reading {data_name}")
    trans_lines_gdf = gpd.read_file(os.path.join(data_path, data_name))
    trans_lines_gdf.rename({"ID":"line_id"}, inplace=True, axis=1)
    # Retain initial crs
    trans_lines_crs = trans_lines_gdf.crs
    trans_lines_gdf = trans_lines_gdf.to_crs("EPSG:4326")
    # Translate MultiLineString to LineString geometries, taking only the first LineString
    trans_lines_gdf.loc[
    trans_lines_gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
    ] = trans_lines_gdf.loc[
    trans_lines_gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
    ].apply(lambda x: list(x.geoms)[0])
    # Get rid of erroneous 1MV and low power line voltages
    trans_lines_gdf = trans_lines_gdf[(trans_lines_gdf["VOLTAGE"] >= 200)]
    # Plot the lines
    for idx, row in trans_lines_gdf.iterrows():
        x, y = row['geometry'].xy
        ax.plot(x, y, color='black', linewidth=1,transform=transform)

fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)

# Filter out sites with error message
info_df = info_df[~info_df['error'].str.contains('', na=False)]
# TODO: Print number of GIC sites removed due to error and how many kept.

# Remove rows that don't have data_type = GIC and data_class = measured
info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
info_df.reset_index(drop=True, inplace=True)

sites = info_df['site_id'].tolist()
#sites = ['Bull Run'] # For testing

# Read in cc data
pkl_file = os.path.join(data_dir, '_results', 'cc.pkl')
with open(pkl_file, 'rb') as file:
  print(f"Reading {pkl_file}")
  cc_rows = pickle.load(file)
cc_df = pd.DataFrame(cc_rows)
cc_df.reset_index(drop=True, inplace=True)


cc_vs_dist_map(cc_df)
std_map(info_df, cc_df)
site_plots(info_df, cc_df)
#transmission_map() # Content of transmission_map.py
#location_map() # Content of map.py

# then rename this file to plot_maps.py and delete transmission_map.py and cc_map.py



