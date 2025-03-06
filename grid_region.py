import os
import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

# Read the GeoJSON file
geojson_file = os.path.join('..', '2024-AGU-data', 'nerc', 'nerc_gdf.geojson')
print(f"Reading {geojson_file}")
gdf = gpd.read_file(geojson_file)

# Plot the GeoDataFrame with colors for each region
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf.plot(column='REGIONS', ax=ax, legend=True, legend_kwds={'bbox_to_anchor': (1, 1)}, cmap='tab20')
plt.title('Regions Map')
#plt.show()
plt.close()

# Function to map power pool regions to US regions
def map_region_to_pool(region):
    if region == 'ERCOT':
        return 'ERCOT'
    elif region in ['PJM', 'NYISO', 'ISONE', 'SCRTP', 'SERTP', 'FRCC']:  # Add all eastern regions here
        return 'East'
    elif region in ['WestConnectNonEnrolled', 'WestConnect', 'CAISO', 'NorthernGridConnected', 'NorthernGridUnconnected', 'NotOrder1000']:  # Add all western regions here
        return 'West'
    elif region in ['SPP', 'MISO']:  # Add all western regions here
        return 'Central'
    else:
        return 'Unknown'  # Default case if region is not found

# Add a new column to the GeoDataFrame for the power pool category
gdf['US_region'] = gdf['REGIONS'].apply(map_region_to_pool)

# Plot the GeoDataFrame with colors for each US region
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf.plot(column='US_region', ax=ax, legend=True, legend_kwds={'bbox_to_anchor': (1, 1)}, cmap='tab20')
plt.title('US Regions Map')
#plt.show()
plt.close()

# Save the updated GeoDataFrame to a CSV file
csv_file = os.path.join('..', '2024-AGU-data', 'nerc', 'nerc_gdf_mapped.csv')
print(f"Saving {csv_file}")
gdf.to_csv(csv_file, index=False)

# Reading in info.extended.csv
fname = os.path.join('info', 'info.extended.csv')
print(f"Reading {fname}")
info_df = pd.read_csv(fname)
info_gdf = gpd.GeoDataFrame(info_df, geometry=gpd.points_from_xy(info_df.geo_lon, info_df.geo_lat))
info_gdf = info_gdf.drop(columns=['power_pool', 'US_region'], errors='ignore')

# Spatial join to assign REGION and US_region to the locations in info_df
info_gdf = gpd.sjoin(info_gdf, gdf[['REGIONS', 'US_region', 'geometry']], how='left', predicate='intersects')
# Rename columns
info_gdf = info_gdf.rename(columns={'REGIONS': 'power_pool'})
# Drop index_right column
info_gdf = info_gdf.drop(columns=['index_right'])

# Convert back to DataFrame
info_df = pd.DataFrame(info_gdf.drop(columns='geometry'))

# Save the updated DataFrame 
out_fname = os.path.join('info', 'info.extended.csv')
info_df.to_csv(out_fname, index=False)
print(f"Saving updated {out_fname}")





