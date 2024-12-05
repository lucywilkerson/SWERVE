import os
import csv

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches

data_dir = '../2024-AGU-data'
projection = ccrs.Miller()
crs = ccrs.PlateCarree()
transform = ccrs.PlateCarree()
state = True # Show political boundaries

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

def add_symbols(ax, coords, val, source, transform, markersize):

    for i in range(len(coords)):
        coord=coords[i]
        if source[i] == 'TVA':
            mark='.'
            if val[i] == 'GIC, predicted':
                mark='^'
        elif source[i] == 'NERC':
            mark='+'
        else:
            continue
        if val[i] == 'GIC, measured':
            col='b'
            face='none'
        elif val[i] == 'GIC, predicted':
            col='c'
            face='none'
        elif val[i] == 'B, measured':
            col='r'
            face=col
        else:
            continue
        ax.plot(coord[1], coord[0], mfc=face, marker=mark, color=col, markersize=markersize, transform=transform)

name=[]
coords=[]
val=[]
source=[]
fname = os.path.join(data_dir, 'Data_Sources.csv')
print(f"Reading {fname}")
with open(fname, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    header = next(plots)
    #print(header)
    for row in plots:
        name.append(row[0]) #adding to name array
        coords.append((float(row[1]),float(row[2]))) #adding to coordinate array
        val.append(row[3]) #adding to value type array
        source.append(row[4]) #adding to data source array
        #print(', '.join(row))

# Create a figure and axes with a specific projection
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})

add_features(ax, state)

# Set the extent of the map (USA)
ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)

add_symbols(ax, coords, val, source, transform, 5)

# TVA region
ax.add_patch(patches.Rectangle([-91, 33], 9, 5, fc='lightyellow', ec='g', transform=transform))
fname = os.path.join(data_dir, 'map', 'map')
savefig(fname)

# Create a figure for just TVA
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})

# Set the extent of the map (TVA)
ax.set_extent([-91, -82, 33, 38], crs=crs)

add_symbols(ax, coords, val, source, transform, 13)

ax.add_patch(patches.Rectangle([-91, 33], 9, 5, fc='lightyellow', ec='g', transform=transform))
add_features(ax, state)

fname = os.path.join(data_dir, 'map', 'map_zoom_tva')
savefig(fname)
