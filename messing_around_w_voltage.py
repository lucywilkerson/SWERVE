import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy.ma as ma
import numpy as np

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

"""
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


plt.scatter(plot_volts, voltage_counts, color='c')
plt.bar(plot_volts, voltage_counts, color = 'c', width=3)
plt.xlabel('Voltage [kV]')
plt.ylabel('Number of Lines')
plt.title('Number of Transmission Lines by Voltage')
plt.yscale('log')
plt.grid()
plt.show()
"""

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
# Set the extent of the map (TVA)
ax.set_extent([-91, -82, 33, 38], crs=crs)


for voltage in voltages:
    if voltage < 200 or voltage > 765:
        continue
    trans_lines_plot = trans_lines_gdf[(trans_lines_gdf["VOLTAGE"] == voltage)]
    # Plot the lines
    if voltage == 765:
        color = 'c'
    elif voltage == 500:
        color = 'm'
    elif voltage == 345:
        color = 'y'
    #elif voltage == 230:
        #color = 'b'
    else:
        continue
    legend_switch = True
    for idx, row in trans_lines_plot.iterrows():
        x, y = row['geometry'].xy
        if legend_switch:
            label = f"{voltage} kV"
            legend_switch = False
        else:
            label = None
        ax.plot(x, y, color=color, linewidth=1,transform=transform, label=label) 


#adding locations!!
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

TVA_sites = ['Bull Run', 'Montgomery', 'Union', 'Widows Creek'] # For testing
df = info_df[info_df['site_id'].isin(TVA_sites)]

add_symbols(ax, df, transform, 13)

ax.legend(loc='upper left')

fname = 'trans_lines_TVA'
out_dir = os.path.join('..', '2024-AGU-data', '_results')
fname = os.path.join(out_dir, fname)
plt.savefig(f'{fname}.png', bbox_inches='tight')

plt.show() 
plt.close()

###############################################################################################################

# comparison plots!

def read_TVA_or_NERC(row):
  site_id = row['site_id']
  data_dir = os.path.join('..', '2024-AGU-data', 'processed')
  if row['data_source'] == 'NERC':
      #reading in data for site if NERC
      fname = os.path.join(data_dir, site_id, 'GIC_measured_NERC.pkl')
  elif row['data_source'] == 'TVA':
      #reading in data for site if TVA
      site_id = "".join(site_id.split()) #removing space from name to match file name
      fname = os.path.join(data_dir, site_id, 'GIC_measured_TVA.pkl')

  with open(fname, 'rb') as f:
      #print(f"Reading {fname}")
      site_data = pickle.load(f)

  site_df = pd.DataFrame(site_data)
  time = site_df['modified'][0]['time']
  mod_data = site_df['modified'][0]['data'] # 1-min avg data
  masked_data = ma.masked_invalid(mod_data) # 1-min data w nan values masked
  return time, mod_data, masked_data


rows = []
for idx_1, row in info_df.iterrows():

  site_1_id = row['site_id']
  if site_1_id not in TVA_sites:
    continue

  site_1_time, site_1_data, msk_site_1_data = read_TVA_or_NERC(row)

  for idx_2, row in info_df.iterrows():
    if idx_1 <= idx_2:  # Avoid duplicate pairs
      continue

    site_2_id = row['site_id']

    if site_2_id not in TVA_sites:
      continue

    site_2_time, site_2_data, msk_site_2_data = read_TVA_or_NERC(row)

    #plotting!!
    plt.figure()
    error_shift = 70
    yticks = np.arange(-120, 30, 10)
    labels = []
    for ytick in yticks:
        if ytick < -30:
            labels.append(str(ytick+error_shift))
        else:
            labels.append(str(ytick))
    kwargs = {"color": 'w', "linestyle": '-', "linewidth": 10, "xmin": 0, "xmax": 1}
    plt.axhline(y=-35, **kwargs)
    plt.axhline(y=-120, **kwargs)
    plt.title(f'{site_1_id} vs {site_2_id} GIC Comparison')
    plt.grid()
    plt.plot()
    plt.plot(site_1_time, site_1_data, label=site_1_id, linewidth=0.5)
    plt.plot(site_2_time, site_2_data, label=site_2_id, linewidth=0.5)
    plt.plot(site_1_time, site_1_data-site_2_data-error_shift, color=3*[0.3], label='difference', linewidth=0.5)
    plt.legend()
    plt.ylabel('[A]', rotation=0, labelpad=10)
    plt.ylim(-120, 30)
    plt.yticks(yticks, labels=labels)
    fname = f'{site_1_id}_{site_2_id}_GIC_compare_timeseries'
    out_dir = os.path.join('..', '2024-AGU-data', '_results')
    fname = os.path.join(out_dir, fname)
    plt.savefig(f'{fname}.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


