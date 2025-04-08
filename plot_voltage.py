import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from shapely.geometry import Point
import numpy as np



# getting data as in cc_map.py
data_dir = os.path.join('..', '2024-May-Storm-data')
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

# Read in cc data
pkl_file = os.path.join(data_dir, '_results', 'cc.pkl')
with open(pkl_file, 'rb') as file:
  print(f"Reading {pkl_file}")
  cc_rows = pickle.load(file)
cc_df = pd.DataFrame(cc_rows)
cc_df.reset_index(drop=True, inplace=True)

def savefig(fdir, fname, fmts=['png']):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = os.path.join(fdir, fname)

    for fmt in fmts:
        print(f"    Saving {fname}.{fmt}")
        if fmt == 'png':
            plt.savefig(f'{fname}.{fmt}', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

# US Transmission lines
data_path = os.path.join(data_dir, 'Electric__Power_Transmission_Lines')
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
# trans_lines_gdf = trans_lines_gdf[(trans_lines_gdf["VOLTAGE"] >= 200)]
voltages = trans_lines_gdf["VOLTAGE"].unique()
#order voltages from lowest to highest
voltages = sorted(voltages)


voltage_counts = []
plot_volts = []
# counting number of lines for each voltage
for voltage in voltages:
    if voltage < 0:
        continue
    plot_volts.append(voltage)
    voltage_gdf = trans_lines_gdf[(trans_lines_gdf["VOLTAGE"] == voltage)]
    length = len(voltage_gdf)
    print(f"Voltage: {voltage} kV, Number of Lines: {length}")
    voltage_counts.append(length)

plt.figure(figsize=(12, 6))
plt.scatter(plot_volts, voltage_counts, color='k', zorder=2)
for voltage, count in zip(plot_volts, voltage_counts):
    plt.vlines(voltage, 0, count, color='k')
plt.xlabel('Voltage [kV]')
plt.ylabel('Number of Lines')
plt.title(f'US Transmission Lines from HIFLD: {len(trans_lines_gdf)}')
plt.yscale('log')
plt.xscale('log')
plt.grid(True, zorder=1)
out_dir = os.path.join(data_dir, '_results')
fname = os.path.join(out_dir, 'trans_lines_count')
plt.savefig(f'{fname}.png', dpi=600, bbox_inches='tight')
#plt.show()
plt.close()

# Finding line lengths in km
trans_lines_gdf['length_km'] = trans_lines_gdf['geometry'].length * (111.32)

# Plotting length of line as a function of line voltage
voltage_lengths = []
plot_volts = []

for voltage in voltages:
    if voltage < 0:
        continue
    plot_volts.append(voltage)
    voltage_gdf = trans_lines_gdf[(trans_lines_gdf["VOLTAGE"] == voltage)]
    total_length = voltage_gdf['length_km'].sum()
    print(f"Voltage: {voltage} kV, Total Length: {total_length:.2f} km")
    voltage_lengths.append(total_length)

plt.figure(figsize=(12, 6))
plt.scatter(plot_volts, voltage_lengths, color='k', zorder=2)
for voltage, length in zip(plot_volts, voltage_lengths):
    plt.vlines(voltage, 0, length, color='k')
plt.xlabel('Voltage [kV]')
plt.ylabel('Total Length [km]')
plt.title('Length of US Transmission Lines by Voltage')
plt.yscale('log')
plt.xscale('log')
plt.grid(True, zorder=1)
out_dir = os.path.join(data_dir, '_results')
fname = os.path.join(out_dir, 'trans_lines_length')
plt.savefig(f'{fname}.png', dpi=600, bbox_inches='tight')
#plt.show()
plt.close()

####################################################################################################
# this was my code to find the nearest line, will be using Dennies' code (voltage_geography.py) instead
####################################################################################################
"""
# Function to find the nearest line voltage for a given point
def find_nearest_line(point, lines_gdf):
    # Calculate the distance from the point to each line
    projected_lines_gdf = lines_gdf.to_crs(epsg=3857)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    projected_point = point_gdf.to_crs(epsg=3857).geometry.iloc[0]
    distances = projected_lines_gdf.geometry.distance(projected_point)
    # Find nearest line
    nearest_idx = distances.dropna().idxmin()
    nearest_line = trans_lines_gdf.loc[nearest_idx]
    return nearest_line

# Create gdf from info_df
info_gdf = gpd.GeoDataFrame(
    info_df, geometry=gpd.points_from_xy(info_df['geo_lon'], info_df['geo_lat']), crs="EPSG:4326"
)

# Remove lines with negative voltage
trans_lines_gdf = trans_lines_gdf[(trans_lines_gdf["VOLTAGE"] >= 0)]

# Loop over info_gdf
for i in range(len(info_gdf)):
    nearest_line = find_nearest_line(info_gdf['geometry'][i], trans_lines_gdf)
    nearest_voltage = nearest_line['VOLTAGE']
    nearest_length = nearest_line['length_km']
    line_coords = list(nearest_line['geometry'].coords)
    x_diff = line_coords[-1][0] - line_coords[0][0]
    y_diff = line_coords[-1][1] - line_coords[0][1]
    nearest_orientation = np.degrees(np.arctan2(y_diff, x_diff)) # need to consider curvature of Earth?
    # Updating info_df
    info_df.at[i, 'nearest_voltage'] = nearest_voltage
    info_df.at[i, 'nearest_length_km'] = nearest_length
    info_df.at[i, 'nearest_orientation'] = nearest_orientation

# TODO: save this to info.csv, then find differences in site pairs and plot scatter plots"""

