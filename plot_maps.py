import os
import csv
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches

out_dir = os.path.join('..', '2024-AGU-data', 'map')
projection = ccrs.Miller()
crs = ccrs.PlateCarree()
transform = ccrs.PlateCarree()
state = True # Show political boundaries
patch_kwargs = {"fc": 'lightyellow', "ec": 'g', "transform": transform}

def savefig(fname):
  if not os.path.exists(os.path.dirname(fname)):
    os.makedirs(os.path.dirname(fname))

  print(f"Saving {fname}.png")
  plt.savefig(f'{fname}.png', dpi=600, bbox_inches="tight")

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


fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
df = pd.read_csv(fname).set_index('site_id')

def location_map(extent, markersize, out_dir, out_name, patch=False):
  # Create a figure and axes with a specific projection
  fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})
  # Adding map features and symbols for locations
  add_features(ax, state)
  if patch == True:
    ax.add_patch(patches.Rectangle([-91, 33], 9, 5, **patch_kwargs))
  add_symbols(ax, df, transform, markersize)
  # Set the extent of the map
  ax.set_extent(extent, crs=crs)
  # Save map
  fname = os.path.join(out_dir, out_name)
  savefig(fname)
  plt.show()

USA_extent = [-125, -67, 25.5, 49.5]
TVA_extent = [-91, -82, 33, 38]

location_map(USA_extent, 5, out_dir, 'map', patch=True)
location_map(TVA_extent, 13, out_dir, 'map_zoom_TVA', patch=True)