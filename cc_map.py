import os
import pandas as pd
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

data_dir = os.path.join('..', '2024-AGU-data')
projection = ccrs.Miller()
crs = ccrs.PlateCarree()
transform = ccrs.PlateCarree()
state = True # Show political boundaries


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
    return color,mark

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

def map_cc(ax, site_id, cc_df, colors=False, lines=False):
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
        plt.xlabel('Average standard deviation')
    plt.ylabel('|cc|')
    plt.ylim(0, 1)
    plt.title(site_id)
    plt.grid(True)
        

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
#sites = ['Bull Run'] # For testing

# Read in cc data
pkl_file = os.path.join(data_dir, '_results', 'cc.pkl')
with open(pkl_file, 'rb') as file:
  print(f"Reading {pkl_file}")
  cc_rows = pickle.load(file)
cc_df = pd.DataFrame(cc_rows)
cc_df.reset_index(drop=True, inplace=True)

# Plotting maps and cc plots for each site

for idx_1, row in info_df.iterrows():
    site_1_id = row['site_id']
    if site_1_id not in sites:
        continue

    # Create a figure and axes with a specific projection
    fig, axs = plt.subplots(2, figsize=(10, 8), subplot_kw={'projection': projection})
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

    savefig(site_1_id, 'cc_map')
    plt.close()

    # plotting cc vs distance
    fig = plot_cc(site_1_id, cc_df)
    #plt.show()

    savefig(site_1_id, 'cc_plot')
    plt.close()

    # plotting cc vs standard devaition
    fig = plot_cc(site_1_id, cc_df, distance=False)
    #plt.show()

    savefig(site_1_id, 'cc_plot_std')
    plt.close()



# mapping top and bottom site pairs
cc_lims = [0.8,0.6,0.4,0.2,0.01]
# distance limit for cc < 0.4
dist_lim = 1000

# function to draw lines between pairs
# TODO: merge with map_cc
def add_pair_lines(ax, site_1_id, site_2_id, color='k'):
    site_1_lat = info_df.loc[info_df['site_id'] == site_1_id, 'geo_lat'].values[0]
    site_1_lon = info_df.loc[info_df['site_id'] == site_1_id, 'geo_lon'].values[0]

    site_2_lat = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lat'].values[0]
    site_2_lon = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lon'].values[0]

    ax.plot([site_1_lon, site_2_lon], [site_1_lat, site_2_lat],
            color=color, transform=transform)

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
    if cc > cc_lims[0]:
        site_1_id = row['site_1']
        site_2_id = row['site_2']
        add_pair_lines(axs[0,0], site_1_id, site_2_id,color='r')
        num_pairs[0] += 1
    elif cc > cc_lims[1]:
        site_1_id = row['site_1']
        site_2_id = row['site_2']
        add_pair_lines(axs[0,1], site_1_id, site_2_id)
        num_pairs[1] += 1
    elif cc > cc_lims[2]:
        site_1_id = row['site_1']
        site_2_id = row['site_2']
        add_pair_lines(axs[0,2], site_1_id, site_2_id)
        num_pairs[2] += 1
    elif cc > cc_lims[3]:
        if row['dist(km)'] > dist_lim:
            continue
        site_1_id = row['site_1']
        site_2_id = row['site_2']
        add_pair_lines(axs[1,0], site_1_id, site_2_id)
        num_pairs[3] += 1
    elif cc > cc_lims[4]:
        if row['dist(km)'] > dist_lim:
            continue
        site_1_id = row['site_1']
        site_2_id = row['site_2']
        add_pair_lines(axs[1,1], site_1_id, site_2_id)
        num_pairs[4] += 1
    elif cc < cc_lims[4]:
        if row['dist(km)'] > dist_lim:
            continue
        site_1_id = row['site_1']
        site_2_id = row['site_2']
        add_pair_lines(axs[1,2], site_1_id, site_2_id)
        num_pairs[5] += 1
                    
labels = [
    f"Number of site pairs with |cc| > {cc_lims[0]}: {int(num_pairs[0])}",
    f"Number of site pairs with {cc_lims[0]} > |cc| > {cc_lims[1]}: {int(num_pairs[1])}",
    f"Number of site pairs with {cc_lims[1]} > |cc| > {cc_lims[2]}: {int(num_pairs[2])}",
    f"Number of site pairs with {cc_lims[2]} > |cc| > {cc_lims[3]}: {int(num_pairs[3])}",
    f"Number of site pairs with {cc_lims[3]} > |cc| > {cc_lims[4]}: {int(num_pairs[4])}",
    f"Number of site pairs with |cc| < {cc_lims[4]}: {int(num_pairs[5])}"
]

for ax, label in zip(axs.flat, labels):
    ax.text(0.5, -0.1, label, ha="center", transform=ax.transAxes, fontsize=8)
plt.tight_layout()
#plt.show()
output_fname = os.path.join(data_dir, '_results', 'cc_map_pairs')
if not os.path.exists(os.path.dirname(output_fname)):
  os.makedirs(os.path.dirname(output_fname))
print(f"Saving {output_fname}")
plt.savefig(f'{output_fname}.png', bbox_inches='tight')
plt.close()


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
output_fname = os.path.join(data_dir, '_results', 'std_map')
if not os.path.exists(os.path.dirname(output_fname)):
  os.makedirs(os.path.dirname(output_fname))
print(f"Saving {output_fname}")
plt.savefig(f'{output_fname}.png', bbox_inches='tight')
plt.close()
        
