import os
import csv
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches

data_dir = os.path.join('..', '2024-AGU-data')
projection = ccrs.Miller()
crs = ccrs.PlateCarree()
transform = ccrs.PlateCarree()
state = True # Show political boundaries
patch_kwargs = {"fc": 'lightyellow', "ec": 'g', "transform": transform}


def savefig(sid, fname, sub_dir="", fmts=['png']):
  fdir = os.path.join(data_dir, 'processed', sid.lower().replace(' ', ''), sub_dir)
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

def add_cc_colors(cc):  
    if cc < 0.2:
        color = 'red'
    elif cc < 0.4:
        color = 'orange'
    elif cc < 0.6:
        color = 'yellow'
    elif cc < 0.8:
        color = 'lightgreen'
    else:
        color = 'green'
    return color

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

def plot_cc(ax, site_id, cc_df, colors=False, lines=False):
    site_1_lat = info_df.loc[info_df['site_id'] == site_id, 'geo_lat'].values[0]
    site_1_lon = info_df.loc[info_df['site_id'] == site_id, 'geo_lon'].values[0]

    for idx, row in cc_df.iterrows():
        if row['site_1'] == site_id:
            site_2_id = row['site_2']
        elif row['site_2'] == site_id:
            site_2_id = row['site_1']
        else:
            continue
        cc = row['cc']
        site_2_lat = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lat'].values[0]
        site_2_lon = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lon'].values[0]
        if colors == True:
            col = add_cc_colors(cc)
            ax.plot(site_2_lon, site_2_lat, color=col, 
                    marker='o', markersize=5, transform=transform)
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
            
        
        

fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)

# Filter out sites with error message
# Also remove rows that don't have data_type = GIC and data_class = measured
info_df = info_df[~info_df['error'].str.contains('', na=False)]
info_df = info_df[info_df['data_type'].str.contains('GIC', na=False)]
info_df = info_df[info_df['data_class'].str.contains('measured', na=False)]
info_df.reset_index(drop=True, inplace=True)

sites = info_df['site_id'].tolist()
sites = ['Bull Run'] # For testing

# Read in cc data
pkl_file = os.path.join(data_dir, '_results', 'cc.pkl')
with open(pkl_file, 'rb') as file:
  print(f"Reading {pkl_file}")
  cc_rows = pickle.load(file)
cc_df = pd.DataFrame(cc_rows)


for idx_1, row in info_df.iterrows():
    site_1_id = row['site_id']
    if site_1_id not in sites:
        continue

    # Create a figure and axes with a specific projection
    fig, axs = plt.subplots(2, figsize=(10, 8), subplot_kw={'projection': projection})

    for ax in axs.flat:
        add_features(ax, state)
        # Set the extent of the map (USA)
        ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)

    plot_cc(axs[0], site_1_id, cc_df, lines=True)

    plot_cc(axs[1], site_1_id, cc_df, colors=True)

    plt.suptitle(site_1_id)

    

plt.show()