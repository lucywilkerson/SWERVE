import os
import csv
import pandas as pd
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
import geopandas as gpd
from scipy.io import loadmat
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import LineString, box

from storminator import FILES, DATA_DIR, LOG_DIR, plt_config, savefig, savefig_paper, subset, add_subplot_label

import utilrsw
logger = utilrsw.logger(log_dir=LOG_DIR)

results_dir = os.path.join(DATA_DIR, '_results')
out_dir = os.path.join(DATA_DIR, '_map')
info_fname = FILES['info']['extended']
mag_lat_fname = FILES['shape_files']['geo_mag']
beta_fname = FILES['analysis']['beta']
pkl_file = FILES['analysis']['cc']
trans_lines_fname = FILES['shape_files']['electric_power']


projection = ccrs.Miller()
crs = ccrs.PlateCarree()
transform = ccrs.PlateCarree()
state = True # Show political boundaries

# Set types of maps to produce
map_location = True
map_cc = False
map_std = False
map_sites =  False
map_beta = False
map_transmission = False


fmts=['png', 'pdf']
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

def add_symbols(ax, df, transform, markersize):

    for row in df.itertuples():
        # Select the symbol
        symbol_dict = {
            ('TVA', 'GIC', 'calculated'): ('^', 'c', 'none'),
            ('TVA', 'GIC', 'measured'): ('.', 'b', 'none'),
            ('TVA', 'B', 'measured'): ('.', 'r', 'r'),
            ('NERC', 'GIC', 'calculated'): ('+', 'c', 'none'),
            ('NERC', 'GIC', 'measured'): ('+', 'b', 'none'),
            ('NERC', 'B', 'measured'): ('+', 'r', 'r')
        }

        key = (row.data_source, row.data_type, row.data_class)
        symbol = symbol_dict.get(key, None)
        if symbol is None:
            continue

        marker, color, face = symbol

        ax.plot(row.geo_lon, row.geo_lat,
                mfc=face, marker=marker, color=color,
                markersize=markersize,
                transform=transform)

#reading in info.extended.csv
print(f"Reading {info_fname}")
df = pd.read_csv(info_fname).set_index('site_id')
df = df[~df['error'].str.contains('', na=False)] #remove sites w error

# Filter out sites with error message
info_df = pd.read_csv(info_fname)
info_df = info_df[~info_df['error'].str.contains('', na=False)]
# TODO: Print number of GIC sites removed due to error and how many kept.
# Remove rows that don't have data_type = GIC and data_class = measured
info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
info_df.reset_index(drop=True, inplace=True)

sites = info_df['site_id'].tolist()

def location_map(extent, markersize, out_dir, out_name, mag_lat=False, patch=False):
  # Create a figure and axes with a specific projection
  fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})
  # Adding map features and symbols for locations
  add_features(ax, state)
  if patch == True:
    patch_kwargs = {"fc": 'lightyellow', "ec": 'g', "transform": transform}
    ax.add_patch(patches.Rectangle([-91, 33], 9, 5, **patch_kwargs))
  
  if mag_lat == True:
        print(f'Reading {mag_lat_fname}')
        mag_gdf = gpd.read_file(mag_lat_fname)
        mag_gdf = mag_gdf.to_crs("EPSG:4326")
        mag_gdf = mag_gdf[mag_gdf["geometry"].notnull()]
    
        # Plotting lines from geometry
        mag_gdf.loc[
        mag_gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
        ] = mag_gdf.loc[
        mag_gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
        ].apply(lambda x: list(x.geoms)[0])
        for idx, row in mag_gdf.iterrows():
            x, y = row['geometry'].xy
            ax.plot(x, y, color='gray', linewidth=.5, transform=transform)
            text = row['Contour']
            if text <=70 and text >= 52:
                line = row['geometry']
                map_box = box(extent[0], extent[2], extent[1], extent[3])
                intersection = line.intersection(map_box)

                # If intersection is a LineString or MultiLineString, get the endpoint on the east (right) extent
                if intersection.is_empty:
                    continue
                if intersection.geom_type == "MultiLineString":
                    # Take the first line
                    intersection = list(intersection.geoms)[0]
                if intersection.geom_type == "LineString":
                    coords = list(intersection.coords)
                    # Find the point(s) with longitude == east extent
                    east_x = extent[1]
                    east_points = [pt for pt in coords if np.isclose(pt[0], east_x, atol=0.05)]
                    if east_points:
                        endpoint = east_points[0]
                    else:
                        # fallback: picks the endpoint with max longitude
                        endpoint = max(coords, key=lambda pt: pt[0])
                    ax.text(endpoint[0], endpoint[1], text, fontsize=8, transform=transform, ha='right', va='center')
                elif intersection.geom_type == "Point":
                    ax.text(intersection.x, intersection.y, text, fontsize=8, transform=transform, ha='right', va='center')
  
  add_symbols(ax, df, transform, markersize)
  # Set the extent of the map
  ax.set_extent(extent, crs=crs)
  # Save map
  savefig(out_dir, out_name)

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
    savefig(results_dir, 'cc_vs_dist_map')
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
    savefig(results_dir, 'std_map')
    plt.close()

def site_maps(info_df, cc_df):

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
        # saving figure
        sid = site_1_id
        sub_dir=""
        fdir = os.path.join(DATA_DIR, '_processed', sid.lower().replace(' ', ''), sub_dir)
        savefig(fdir, 'cc_vs_dist_map')
        plt.close()

def beta_maps():
    
    beta_site='OTT'

    data = loadmat(beta_fname)
    data = data['waveform'][0]
    
    beta_sites = ['MEA', 'OTT', 'MMB', 'NUR']
    if beta_site not in beta_sites:
        raise ValueError(f"Invalid beta site. Choose from {beta_sites}")

    # Convert the MATLAB data to a pandas DataFrame
    raw_df = pd.DataFrame(data)
    if beta_site == 'MEA':
        betas = raw_df[0][0][0][0][1][0]
    if beta_site == 'OTT':
        betas = raw_df[0][1][0][0][1][0]
    if beta_site == 'MMB':
        betas = raw_df[0][2][0][0][1][0]
    if beta_site == 'NUR':
        betas = raw_df[0][3][0][0][1][0]

    # Convert betas list to a Pandas DataFrame
    rows = []
    for i in range(len(betas)):
      beta = betas[i][0][0][1][0][0]
      lat = betas[i][0][0][2][0][0]
      lon = betas[i][0][0][3][0][0]
      rows.append([beta, lat, lon])
    df = pd.DataFrame(rows, columns=['beta', 'lat', 'lon'])

    # Create the interpolator
    interpolator = LinearNDInterpolator(df[['lat', 'lon']], df['beta'])

    # Define a grid for interpolation
    lat_grid = np.linspace(df['lat'].min(), df['lat'].max(), 100)
    lon_grid = np.linspace(df['lon'].min(), df['lon'].max(), 100)
    lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)

    # Interpolate beta values onto the grid
    beta_grid = interpolator(lat_grid, lon_grid)

    # Plot the interpolated data on a map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max()], crs=ccrs.PlateCarree())

    # Add features to the map
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # Plot the interpolated beta values
    contour = ax.contourf(lon_grid, lat_grid, beta_grid, transform=ccrs.PlateCarree(), cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.05, aspect=50)
    cbar.set_label('Interpolated Beta')

    # Add a title
    ax.set_title('Interpolated Beta Values')

    # Show the plot
    plt.show()

def transmission_map(info_df, gdf, cc_df, std=False):

    def add_trans_lines(ax, gdf):
        gdf = gdf.to_crs("EPSG:4326")
        # Translate MultiLineString to LineString geometries, taking only the first LineString
        gdf.loc[
        gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
        ] = gdf.loc[
        gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
        ].apply(lambda x: list(x.geoms)[0])
        # Get rid of erroneous 1MV and low power line voltages
        gdf = gdf[(trans_lines_gdf["VOLTAGE"] >= 200)]
        # Plot the lines
        for idx, row in gdf.iterrows():
            x, y = row['geometry'].xy
            ax.plot(x, y, color='black', linewidth=1,transform=transform)

    def add_loc(ax, info_df, cc_df, stdev=False):
        # plotting standard deviation
        for idx_1, row_1 in info_df.iterrows():
            site_id = row_1['site_id']
            if site_id not in sites:
                continue
            site_lat = info_df.loc[info_df['site_id'] == site_id, 'geo_lat'].values[0]
            site_lon = info_df.loc[info_df['site_id'] == site_id, 'geo_lon'].values[0]
            if stdev == True:
                for idx_2, row_2 in cc_df.iterrows():
                    if row_2['site_1'] == site_id: #TODO: with new cc.pkl format, this doesn't work
                        std = row_2['std_1']
                    else:
                        continue
                    ax.plot(site_lon, site_lat, color='r', marker='o', markersize=std, transform=transform)
            else:
                ax.plot(site_lon, site_lat, color='r', marker='o', markersize=5, transform=transform)

    # Setting up map
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})
    add_features(ax, state)
    # Set the extent of the map (USA)
    ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)
    # Add transmission lines
    add_trans_lines(ax, gdf)
    # Add GIC sites
    add_loc(ax, info_df, cc_df, stdev=std)
    if std == False:
        ax.set_title(r'US Transmission Lines $\geq$ 200kV w "good" GIC Sites', fontsize=15)
        savefig(results_dir, 'transmission_map')
    elif std == True:
        ax.set_title(r'US Transmission Lines $\geq$ 200kV w GIC Standard Deviation', fontsize=15)
        savefig(results_dir, 'transmission_std_map')


# Read in cc data
with open(pkl_file, 'rb') as file:
  print(f"Reading {pkl_file}")
  cc_rows = pickle.load(file)
cc_df = pd.DataFrame(cc_rows)
cc_df.reset_index(drop=True, inplace=True)

if map_location:
    USA_extent = [-125, -67, 25.5, 49.5]
    TVA_extent = [-91, -82, 33, 38]

    location_map(USA_extent, 5, out_dir, 'map', mag_lat=True, patch=True)
    location_map(TVA_extent, 13, out_dir, 'map_zoom_TVA', patch=True)
if map_cc:
    cc_vs_dist_map(cc_df)
if map_std:
    std_map(info_df, cc_df)
if map_sites:
    site_maps(info_df, cc_df)
if map_beta:
    beta_maps()
if map_transmission:
    # US Transmission lines
    print(f"Reading {trans_lines_fname}")
    trans_lines_gdf = gpd.read_file(trans_lines_fname)
    trans_lines_gdf.rename({"ID":"line_id"}, inplace=True, axis=1)

    transmission_map(info_df, trans_lines_gdf, cc_df)
    transmission_map(info_df, trans_lines_gdf, cc_df, std=True)


exit()
##################################################################
# stuff from messing w voltage (ie just TVA)

# Translating geometries
trans_lines_gdf = trans_lines_gdf.to_crs("EPSG:4326")
# Translate MultiLineString to LineString geometries, taking only the first LineString
trans_lines_gdf.loc[
trans_lines_gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
] = trans_lines_gdf.loc[
trans_lines_gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
].apply(lambda x: list(x.geoms)[0])

# Setting up map
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})
add_features(ax, state)
# Set the extent of the map (TVA)
ax.set_extent(TVA_extent, crs=crs)

voltages = trans_lines_gdf["VOLTAGE"].unique()
#order voltages from lowest to highest
voltages = sorted(voltages)

for voltage in voltages:
    if voltage < 200 or voltage > 765:
        continue
    trans_lines_plot = trans_lines_gdf[(trans_lines_gdf["VOLTAGE"] == voltage)]
    # Plot the lines
    if voltage == 765:
        color = 'r'
    elif voltage == 500:
        color = 'g'
    elif voltage == 345:
        color = 'b'
    #elif voltage == 230:
        #color = 'y'
    else:
        continue
    legend_switch = True
    for idx, row in trans_lines_plot.iterrows():
        x, y = row['geometry'].xy
        if legend_switch:
            label = f"{voltage} kV"
            legend_switch = False
        else:
            label = None
        ax.plot(x, y, color=color, linewidth=2,transform=transform, label=label) 


#adding locations!!

TVA_sites = ['Bull Run', 'Montgomery', 'Union', 'Widows Creek'] # For testing
TVA_df = info_df[info_df['site_id'].isin(TVA_sites)]

add_symbols(ax, TVA_df, transform, 13)

ax.legend(loc='upper left')

fname = 'trans_lines_TVA'
out_dir = os.path.join('..', '2024-May-Storm-data', '_results')
fname = os.path.join(out_dir, fname)
plt.savefig(f'{fname}.png', dpi=600, bbox_inches='tight') 
plt.close()
