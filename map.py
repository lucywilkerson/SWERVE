import os
import csv
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches

data_dir = '../2024-AGU-data'
projection = ccrs.Miller()
crs = ccrs.PlateCarree()
transform = ccrs.PlateCarree()
state = True # Show political boundaries
patch_kwargs = {"fc": 'lightyellow', "ec": 'g', "transform": transform}

def savefig(fname):
  print(f"Saving {fname}.png")
  plt.savefig(f'{fname}.png', dpi=600, bbox_inches="tight")
  #plt.savefig(f'{fname}.pdf')
  #plt.savefig(f'{fname}.svg')

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

def add_symbols(ax, coords, data_type, data_class, data_source, transform, markersize):

    for i in range(len(coords)):
        symbol_dict = {
            ('TVA', 'GIC', 'calculated'): ('^', 'c', 'none'),
            ('TVA', 'GIC', 'measured'): ('.', 'b', 'none'),
            ('NERC', 'GIC', 'calculated'): ('+', 'c', 'none'),
            ('NERC', 'GIC', 'measured'): ('+', 'b', 'none'),
            ('NERC', 'B', 'measured'): ('+', 'r', 'r')
        }

        key = (data_source[i], data_type[i], data_class[i])
        symbol = symbol_dict.get(key, None)
        if symbol is None:
            continue

        marker, color, face = symbol

        ax.plot(coords[i][1], coords[i][0],
                mfc=face, marker=marker, color=color,
                markersize=markersize,
                transform=transform)


fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
df = pd.read_csv(fname).set_index('site_id')
#coords = (zip(df['latitude'], df['longitude']))
#data_type = df['data_type']
#data_class = df['data_class']
#data_source = df['data_source']

# Create a figure and axes with a specific projection
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})

add_features(ax, state)

# Set the extent of the map (USA)
ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)

add_symbols(ax, zip(df['latitude'], df['longitude']), df['data_type'], df['data_class'], df['data_source'], transform, 5)

# TVA region
ax.add_patch(patches.Rectangle([-91, 33], 9, 5, **patch_kwargs))
fname = os.path.join(data_dir, 'map', 'map')
savefig(fname)

# Create a figure for just TVA
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})

# Set the extent of the map (TVA)
ax.set_extent([-91, -82, 33, 38], crs=crs)

add_symbols(ax, zip(df['latitude'], df['longitude']), df['data_type'], df['data_class'], df['data_source'], transform, 13)

ax.add_patch(patches.Rectangle([-91, 33], 9, 5, **patch_kwargs))
add_features(ax, state)

fname = os.path.join(data_dir, 'map', 'map_zoom_tva')
savefig(fname)
