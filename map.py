import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv
import matplotlib.patches as patches

#reading in locations from csv
name=[]
coords=[]
val=[]
source=[]
with open('Data_Sources.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    header = next(plots)
    print(header)
    for row in plots:
        name.append(row[0]) #adding to name array
        coords.append((float(row[1]),float(row[2]))) #adding to coordinate array
        val.append(row[3]) #adding to value type array
        source.append(row[4]) #adding to data source array
        print(', '.join(row))

#save fig
def plot_save(fname):
  print(f"Saving {fname}")
  plt.savefig(f'{fname}.png',dpi=300)
  plt.savefig(f'{fname}.pdf')
  plt.savefig(f'{fname}.svg')

# Create a figure and axes with a specific projection
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

#switch for state lines
state = True

# Add coastlines and other features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
if state==True:
    ax.add_feature(cfeature.STATES)

# Set the extent of the map (USA)
ax.set_extent([-127, -62, 24, 51], crs=ccrs.PlateCarree())

# Add pins to the map
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
    ax.plot(coord[1], coord[0], marker=mark, color=col, markersize=5, transform=ccrs.PlateCarree())

#adding patch for TVA
ax.add_patch(patches.Rectangle([-91,33],9,5,fc='lightyellow',ec='g'))

#saving
plot_save('data_map')


# Create a figure for just TVA
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

#switch for state lines
state = True

# Add coastlines and other features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
if state==True:
    ax.add_feature(cfeature.STATES)

# Set the extent of the map (TVA)
ax.set_extent([-91, -82, 33, 38], crs=ccrs.PlateCarree())

# Add pins to the map
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
    ax.plot(coord[1], coord[0], marker=mark, color=col, mfc=face, markersize=13, transform=ccrs.PlateCarree())

ax.add_patch(patches.Rectangle([-91,33],9,5,fc='lightyellow',ec='g'))

plot_save('tva_map')
