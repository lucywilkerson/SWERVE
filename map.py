import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv

#reading in locations from csv
name=[]
coords=[]
val=[]
source=[]
with open('Data Sources.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    header = next(plots)
    print(header)
    for row in plots:
        name.append(row[0]) #adding to name array
        coords.append((float(row[1]),float(row[2]))) #adding to coordinate array
        val.append(row[3]) #adding to value type array
        source.append(row[4]) #adding to data source array
        print(', '.join(row))

#create figure and axes with  specific projection
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

#add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

#set the limits of map
ax.set_extent([-130, -60, 20, 55], crs=ccrs.PlateCarree())

#add pins to the map
for i in range(len(coords)):
    coord=coords[i]
    if source[i] == 'TVA':
        mark='.'
    elif source[i] == 'NERC':
        mark='+'
    else:
        continue
    if val[i] == 'GIC, measured':
        col='b'
    elif val[i] == 'GIC, predicted':
        col='c'
    elif val[i] == 'B, measured':
        col='r'
    else:
        continue
    ax.plot(coord[1], coord[0], marker=mark, color=col, markersize=5, transform=ccrs.PlateCarree())
#add title
plt.title("Map of Data Locations")
#plt.legend(loc='lower right')

#show map
plt.show()

