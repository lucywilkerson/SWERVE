import os
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# getting data as in cc_map.py
data_dir = os.path.join('..', '2024-AGU-data')
fname = os.path.join('info', 'info.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)
# Filter out sites with error message
info_df = info_df[~info_df['error'].str.contains('', na=False)]
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

# US Transmission lines
data_path = os.path.join('..', '2024-AGU-data', 'Electric__Power_Transmission_Lines')
trans_lines_gdf = gpd.read_file(os.path.join(data_path, 'Electric__Power_Transmission_Lines.shp'))
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

#######################################################################################################################
 
# base map
state = True # Show political boundaries
transform = ccrs.PlateCarree()
projection = ccrs.Miller()
crs = ccrs.PlateCarree()
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
fig, ax = plt.subplots(1, figsize=(10, 8), subplot_kw={'projection': projection})
add_features(ax, state)
# Set the extent of the map (USA)
ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)

# Plot the lines
#fig, ax = plt.subplots(1, 1, figsize=(15, 15))
trans_lines_gdf.plot(ax=ax, edgecolor='black')

# adding points for the sites to geodataframe
site_gdf = gpd.GeoDataFrame(
    info_df, geometry=gpd.points_from_xy(info_df['geo_lon'], info_df['geo_lat']))
site_gdf.crs = "EPSG:4326"

# plotting the sites
site_gdf.plot(ax=ax, marker='o', color='red')
 
# Set the Title
ax.set_title('Transmission Lines within the Contigous US', fontsize=15)
ax.grid(True)
 
# Adding the base map
#ctx.add_basemap(ax, crs=trans_lines_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron, attribution="")

# Remove x and y axis ticks
ax.set_xticks([])
ax.set_yticks([])
 
#plt.show()
plt.close()

#######################################################################################################################
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})
add_features(ax, state)
#ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)
for idx, row in trans_lines_gdf.iterrows():
    x, y = row['geometry'].xy
    ax.plot(x, y, color='black', linewidth=1)
ax.plot(site_gdf['geometry'].x,site_gdf['geometry'].y, 'ro', markersize=3)
plt.show()
