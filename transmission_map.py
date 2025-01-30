import os
import geopandas as gpd
import matplotlib.pyplot as plt
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

## Function to add transmission lines to the map ##
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

# Setting up map
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})
add_features(ax, state)
# Set the extent of the map (USA)
ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)

add_trans_lines(ax)

# adding points for the sites to geodataframe
site_gdf = gpd.GeoDataFrame(
    info_df, geometry=gpd.points_from_xy(info_df['geo_lon'], info_df['geo_lat']))
site_gdf.crs = "EPSG:4326"

# plotting the sites
ax.plot(site_gdf['geometry'].x,site_gdf['geometry'].y, 'ro', markersize=3,transform=transform)
 
# Set the Title
ax.set_title('Transmission Lines within the Contigous US', fontsize=15)
 
plt.show()
plt.close()

