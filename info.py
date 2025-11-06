import os
import json
import pandas as pd
import numpy as np

from swerve import config, subset, infodf2dict

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

"""
Write info/info.extended.csv with additional columns from info/info.csv:
  * interpolated_beta
  * log10_beta
  * mag_lat
  * mag_lon
  * alpha
  * nearest_sim_site
  * nearest_volt
  * power_pool
  * US_region

Write info/info.json, which is a dict with keys of site_id and values of
a dict with all measured and calculated data for that site.
"""

# Code for interpolated beta and log10 beta
def add_beta(info_df, beta_fname, beta_site='OTT'):

  from scipy.io import loadmat
  from scipy.interpolate import LinearNDInterpolator

  logger.info(f"Adding interpolated {beta_site} beta and log10 beta columns to extended info_df")
  logger.info(f"  Reading {beta_fname}")
  data = loadmat(beta_fname)
  data = data['waveform'][0]

  beta_sites = ['MEA', 'OTT', 'MMB', 'NUR']
  if beta_site not in beta_sites:
    raise ValueError(f"  Invalid beta {beta_site}. Choose from {beta_sites}")

  # Convert the MATLAB data to a pandas DataFrame
  raw_df = pd.DataFrame(data)
  if beta_site == 'MEA':
    betas = raw_df[0][0][0][0][1][0]
  if beta_site == 'OTT':
    betas = raw_df[0][1][0][0][1][0]
  if beta_site == 'MMB':
    betas = raw_df[0][2][0][0][1][0]
  if beta_site == 'NUR':
    betas = raw_df[0][3][0][0][1][0]

  # Convert betas list to a Pandas DataFrame
  rows = []
  for i in range(len(betas)):
      beta = betas[i][0][0][1][0][0]
      lat = betas[i][0][0][2][0][0]
      lon = betas[i][0][0][3][0][0]
      rows.append([beta, lat, lon])
  df = pd.DataFrame(rows, columns=['beta', 'lat', 'lon'])

  # Create the interpolator
  interpolator = LinearNDInterpolator(df[['lat', 'lon']], df['beta'])

  # Define a grid for interpolation
  lat_grid = np.linspace(df['lat'].min(), df['lat'].max(), 100)
  lon_grid = np.linspace(df['lon'].min(), df['lon'].max(), 100)
  lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)

  # Add a new column to info_df based on the interpolated beta values
  info_df['interpolated_beta'] = interpolator(info_df['geo_lat'], info_df['geo_lon'])
  info_df['log10_beta'] = np.log10(info_df['interpolated_beta'])
  if info_df['interpolated_beta'].isna().any():
    nans = info_df[info_df['interpolated_beta'].isna()]
    for index, row in nans.iterrows():
      logger.warning(f"    Warning: Could not interpolate beta for site {row['site_id']} at ({row['geo_lat']},{row['geo_lon']})")

# Code for geomag coords and alpha
def add_geomag(info_df, date):
  from spacepy.time import Ticktock
  import spacepy.coordinates as coord
  
  def get_geomag_coords(row):
      Re = 6371.0 #mean Earth radius in km
      alt = 0 #altitude in km, can be set to 0 for surface
      # Geomagnetic coordinate doesn't depend much on altitude, and there is ambiguity in definition of Re
      c = coord.Coords([[(alt+Re)/Re, row['geo_lat'], row['geo_lon']]], 'GEO', 'sph', ['Re', 'deg', 'deg'])
      c.ticks = date
      c = c.convert('MAG', 'sph')
      return c.data[0][1], c.data[0][2]  # mag_lat, mag_lon

  logger.info(f"Adding geomagnetic coordinates and alpha on date {date} to extended info_df")

  date = Ticktock([date], 'UTC')

  # Test to check geomag calculation, compare values from this function to results from online calculator at one site
  row = info_df.iloc[0]
  test_stat = get_geomag_coords(row)
  if row['site_id'] == '10052':
    calc_stat = 53.16, -28.995 #calculated via https://geomag.bgs.ac.uk/data_service/models_compass/coord_calc.html 
    dist = np.sqrt((test_stat[0] - calc_stat[0])**2 + (test_stat[1] - calc_stat[1])**2)
    tolerance = .2 # In degrees 
    assert dist <= tolerance, f"Calculated geomagnetic coordinates {test_stat} is not within tolerance of expected {calc_stat}"

  # Applying the function to create new columns
  info_df[['mag_lat', 'mag_lon']] = info_df.apply(lambda row: pd.Series(get_geomag_coords(row)), axis=1)

  # Calculate alpha and add to info_df
  alpha = .001*np.exp(.115*info_df['mag_lat']) #from eqn 3 https://www.nerc.com/pa/Stand/Reliability%20Standards/TPL-007-3.pdf
  info_df['alpha'] = alpha

# Code for GMU simulation sites
def add_sim_site(info_df, sim_file, update_csv=False):

  """
  Assumes dirs 'tva' and 'nerc' exist in the same directory as sim_file.
  """
  if not os.path.exists(sim_file):
    logger.warning(f"Simulation file {sim_file} does not exist. Skipping add_sim_site.")
    return
  
  # add new column based on the nearest simulation site
  def add_nearest_sim_site(info_df, output_df, data_source=['tva', 'nerc']):
    #info_df['nearest_sim_site'] = -1
    for site_type in data_source:
      for site in output_df[f'{site_type}_site'].unique():
        df = output_df[output_df[f'{site_type}_site'] == site]
        if not df.empty and f'{site_type}_dist' in df.columns:
          min_dist_row = df.loc[df[f'{site_type}_dist'].idxmin()]
        else:
          continue

        mask = (info_df['data_source'] == 'GMU') & (info_df['site_id'] == min_dist_row[f'{site_type}_site'])
        info_df.loc[mask, 'nearest_sim_site'] = int(min_dist_row['sim_site'])

  logger.info("Adding GMU simulation sites to extended info_df")

  sim_dir = os.path.dirname(sim_file)

  logger.info(f"  Reading: {sim_file}")
  site_df = pd.read_csv(sim_file)

  no_gmu_df = info_df[info_df['data_source'] != 'GMU']
  meas_df = no_gmu_df[no_gmu_df['data_class'] == 'measured']

  # Find associated TVA/NERC sites for all simulation sites
  output_data = []
  for site in site_df['sub_id'].unique():
    tva_df = None
    nerc_df = None
    tva_site = None
    nerc_site = None

    tva_fname = os.path.join(sim_dir, 'tva', f'site_{site}.csv')
    if os.path.exists(tva_fname):
      tva_df = pd.read_csv(tva_fname)
      tva_site = tva_df['site_1_device'][0]
      if tva_site == 'Widows Creek 2':
        tva_site = 'Widows Creek'
      if tva_site == 'Paradise':
        tva_site = 'Paradise 3'

    nerc_fname = os.path.join(sim_dir, 'nerc', f'site_{site}.csv')
    if os.path.exists(nerc_fname):
      nerc_df = pd.read_csv(nerc_fname)
      nerc_site = f'{nerc_df["site_1_device"][0]}'


    if tva_df is not None or nerc_df is not None:
      sim_lat = None
      sim_lon = None
      tva_lat = None
      tva_lon = None
      tva_dist = None
      nerc_lat = None
      nerc_lon = None
      nerc_dist = None

      if tva_df is not None:
        sim_lat = tva_df['sub lat'][0]
        sim_lon = tva_df['sub lon'][0]
        tva_lat = meas_df.loc[meas_df['site_id'] == tva_site, 'geo_lat'].values[0]
        tva_lon = meas_df.loc[meas_df['site_id'] == tva_site, 'geo_lon'].values[0]
        tva_dist = tva_df['site_1_Distance'][0]

      if nerc_df is not None:
        if sim_lat is None:  # If not already set by TVA
          sim_lat = nerc_df['sub lat'][0]
          sim_lon = nerc_df['sub lon'][0]
        nerc_lat = meas_df.loc[meas_df['site_id'] == nerc_site, 'geo_lat'].values[0]
        nerc_lon = meas_df.loc[meas_df['site_id'] == nerc_site, 'geo_lon'].values[0]
        nerc_dist = nerc_df['site_1_Distance'][0]

      output_data.append({
        'sim_site': site,
        'sim_lat': sim_lat,
        'sim_lon': sim_lon,
        'tva_site': tva_site,
        'tva_lat': tva_lat,
        'tva_lon': tva_lon,
        'tva_dist': tva_dist,
        'nerc_site': nerc_site,
        'nerc_lat': nerc_lat,
        'nerc_lon': nerc_lon,
        'nerc_dist': nerc_dist
      })

  output_df = pd.DataFrame(output_data)

  if update_csv: # Add GMU sites to info.csv
    # adding GMU simulation sites to info.csv
    def add_info(info_df, output_df, data_source=['tva','nerc']):
      for site_type in data_source:
        for site in output_df[f'{site_type}_site'].unique():
          df = output_df[output_df[f'{site_type}_site']==site]
          if not df.empty and f'{site_type}_dist' in df.columns:
            min_dist_row = df.loc[df[f'{site_type}_dist'].idxmin()]
          else:
            continue

          new_row = pd.DataFrame([{
            'site_id': min_dist_row[f'{site_type}_site'],
            'geo_lat': min_dist_row[f'{site_type}_lat'],
            'geo_lon': min_dist_row[f'{site_type}_lon'],
            'data_type': 'GIC',
            'data_class': 'calculated',
            'data_source': 'GMU',
            'error': np.nan
          }])
          info_df = pd.concat([info_df, new_row], ignore_index=True)
      return info_df

    new_info_df = add_info(no_gmu_df, output_df)

    output_fname = os.path.join('info', 'info.csv')
    new_info_df.to_csv(output_fname, index=False)
    logger.info(f"Saved updated info table to {output_fname}")

  add_nearest_sim_site(info_df, output_df)

# Code for power pool and US region
def add_power_pool(info_df, geojson_file):

  import matplotlib.pyplot as plt
  from matplotlib.collections import LineCollection
  from matplotlib.transforms import Bbox

  import geopandas as gpd

  # Code for power pool and US region
  logger.info(f"  Reading {geojson_file}")
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
  base_dir = os.path.dirname(geojson_file)
  csv_file = os.path.join(base_dir, 'nerc_gdf_mapped.csv')
  logger.info(f"  Writing {csv_file}")
  gdf.to_csv(csv_file, index=False)

  # Making info_df -> gdf
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

  return info_df

# Code for nearest voltage
def add_voltage(info_df, transmission_fname):

  import matplotlib.pyplot as plt
  from matplotlib.collections import LineCollection
  from matplotlib.transforms import Bbox

  import geopandas as gpd

  import cartopy.crs as ccrs
  import cartopy.feature as cfeature

  # Code for nearest voltage
  def explode_with_unique_line_id(gdf):
      """
      Explode a GeoDataFrame so that each part gets a unique line_id.
      """
      gdf = gdf.reset_index(drop=True).reset_index(names="original_index")
      exploded = gdf.explode(index_parts=True)
      exploded = exploded.reset_index(level=1)
      exploded["new_line_id"] = exploded.apply(
          lambda row: (
              f"{row['line_id']}_{row['level_1']}"
              if row["level_1"] > 0
              else row["line_id"]
          ),
          axis=1,
      )
      exploded = exploded.drop(columns=["level_1", "line_id", "original_index"])
      exploded = exploded.rename(columns={"new_line_id": "line_id"})
      return exploded

  def load_transmission_lines(tl_loc, crs_target="EPSG:4326"):
      """
      Load and reproject transmission lines.
      """
      trans_lines_gdf = gpd.read_file(tl_loc)
      trans_lines_gdf.rename({"ID": "line_id"}, inplace=True, axis=1)
      trans_lines_gdf = trans_lines_gdf.to_crs(crs_target)
      return trans_lines_gdf

  def filter_transmission_lines(gdf, extent):
      """
      Filter transmission lines within a specified bounding box.
      """
      bounds = gdf.geometry.bounds
      mask = (
          (bounds.minx >= extent["minx"])
          & (bounds.maxx <= extent["maxx"])
          & (bounds.miny >= extent["miny"])
          & (bounds.maxy <= extent["maxy"])
      )
      return gdf[mask]

  def load_devices(crs="EPSG:4326"):
      """
      Create GeoDataFrame based on content of info_df.
      """
      device_df = info_df.copy()
      device_df.rename(
          columns={
              "site_id": "device",
              "geo_lat": "latitude",
              "geo_lon": "longitude",
              "data_type": "data_type",
              "data_class": "data_class",
              "data_source": "data_source",
              "error": "error",
          },
          inplace=True,
      )
      # Filter out only geographic coordinates # only relevant for magnetometers
      #device_df = device_df[device_df["orientation"] == "1 - Geographic"].copy()

      gdf = gpd.GeoDataFrame(
          device_df,
          geometry=gpd.points_from_xy(device_df.longitude, device_df.latitude),
          crs=crs,
      )

      return gdf

  def build_connection_dicts(tl_gdf_subset):
      """
      Build cascaded dictionaries:
        - device_to_lines: {device: [line, ...]}
        - line_to_device: {line: [device, ...]}
        - device_to_line_voltages: {device: [voltage, ...]}
        - device_2_device: {device: [other devices sharing a line, ...]}
        - line_to_device_map: same as line_to_device (for convenience)
      """
      device_to_lines = {}
      line_to_device = {}
      device_to_line_voltages = {}
      device_2_device = {}
      line_to_device_map = {}

      for _, row in tl_gdf_subset.iterrows():
          device_id = row["device"]
          line_id = row["LINE_ID"]
          line_voltage = row["LINE_VOLTAGE"]

          device_to_lines.setdefault(device_id, [])
          if line_id not in device_to_lines[device_id]:
              device_to_lines[device_id].append(line_id)

          device_to_line_voltages.setdefault(device_id, [])
          device_to_line_voltages[device_id].append(line_voltage)

          line_to_device.setdefault(line_id, [])
          if device_id not in line_to_device[line_id]:
              line_to_device[line_id].append(device_id)

      # Build device-to-device connections based on shared lines
      for device, lines in device_to_lines.items():
          connected = set()
          for line in lines:
              for other in line_to_device.get(line, []):
                  if other != device:
                      connected.add(other)
          device_2_device[device] = list(connected)

      line_to_device_map = line_to_device
      return (
          device_to_lines,
          line_to_device,
          device_to_line_voltages,
          device_2_device,
          line_to_device_map,
      )

  def process_data(fname, buffer_distance=1000):
      """
      Process transmission lines and device data.

      Parameters:
        buffer_distance: Buffer (in meters) to apply around devices.

      Returns:
        tl_gdf_subset: GeoDataFrame of transmission lines (subset after spatial join).
        device_gdf: GeoDataFrame of devices with representative points.
        connection_dicts: A tuple of dictionaries:
            (device_to_lines, line_to_device, device_to_line_voltages, device_2_device, line_to_device_map)
      """
      # Define data locations
      tl_loc = fname

      # Load transmission lines and filter by US extent
      trans_lines_gdf = load_transmission_lines(tl_loc, crs_target="EPSG:4326")
      us_extent = {"minx": -125.0, "maxx": -66.9, "miny": 24.4, "maxy": 49.4}
      trans_lines_gdf = filter_transmission_lines(trans_lines_gdf, us_extent)

      # Explode lines for unique IDs and compute length
      trans_lines_gdf = explode_with_unique_line_id(trans_lines_gdf)
      trans_lines_gdf["length"] = trans_lines_gdf.geometry.apply(lambda x: x.length)

      # Load devices
      device_gdf = load_devices(crs="EPSG:4326")

      # Reproject both GeoDataFrames to a projected CRS (NAD83)
      projected_crs = "EPSG:5070"
      device_gdf = device_gdf.to_crs(projected_crs)
      trans_lines_gdf = trans_lines_gdf.to_crs(projected_crs)

      # Buffer devices and create a buffered GeoDataFrame
      device_gdf["buffered"] = device_gdf.geometry.buffer(buffer_distance)
      buffered_gdf = gpd.GeoDataFrame(device_gdf, geometry="buffered")

      # Spatial join: Find transmission lines intersecting device buffers
      intersection_gdf = gpd.sjoin(
          trans_lines_gdf, buffered_gdf, how="inner", predicate="intersects"
      )

      # Process transmission lines GeoDataFrame from the join
      tl_gdf = intersection_gdf.copy()
      tl_gdf = tl_gdf.rename(columns={"geometry_left": "geometry"})
      tl_gdf = tl_gdf.drop(columns=["geometry_right"])
      tl_gdf = tl_gdf.set_geometry("geometry")

      renamed_columns = {
          "line_id": "LINE_ID",
          "geometry": "geometry",
          "NAICS_CODE": "LINE_NAICS_CODE",
          "VOLTAGE": "LINE_VOLTAGE",
          "VOLT_CLASS": "LINE_VOLT_CLASS",
          "INFERRED": "INFERRED",
          "SUB_1": "SUB_1",
          "SUB_2": "SUB_2",
          "length": "LINE_LEN",
          "device": "device",
      }

      tl_gdf_subset = tl_gdf[list(renamed_columns.keys())]
      tl_gdf_subset = tl_gdf_subset.rename(columns=renamed_columns)

      # Process device GeoDataFrame from the join
      device_gdf = intersection_gdf.copy()
      device_gdf = device_gdf.drop(columns=["geometry"])
      device_gdf = device_gdf.rename(columns={"geometry_right": "geometry"})
      device_gdf = device_gdf.set_geometry("geometry")
      device_gdf_subset = device_gdf[list(renamed_columns.keys())]
      device_gdf_subset = device_gdf_subset.rename(columns=renamed_columns)
      device_gdf = gpd.GeoDataFrame(device_gdf_subset, geometry="geometry")

      # Compute representative point (centroid) for devices
      device_gdf["rep_point"] = device_gdf.geometry.centroid
      device_gdf = device_gdf.set_geometry("rep_point")
      device_gdf.crs = tl_gdf_subset.crs

      # Build connection dictionaries
      connection_dicts = build_connection_dicts(tl_gdf_subset)

      return tl_gdf_subset, device_gdf, connection_dicts

  # Haversine distance between two locs
  def haversine_dist(lat1, lon1, lat2, lon2):
      R = 6371
      lat1_rad = np.radians(lat1)
      lon1_rad = np.radians(lon1)
      lat2_rad = np.radians(lat2)
      lon2_rad = np.radians(lon2)
      dlat = lat2_rad - lat1_rad
      dlon = lon2_rad - lon1_rad
      a = (
          np.sin(dlat / 2) ** 2
          + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
      )
      c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
      return R * c

  # Plot the data and the filtered transmission lines
  # set up the cartopy
  def setup_map(ax, spatial_extent=[-125, -66.5, 24, 50]):
      ax.set_extent(spatial_extent, ccrs.PlateCarree())

      ax.add_feature(cfeature.LAND, facecolor="#F0F0F0")
      ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="grey")
      ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#E6F3FF")
      ax.add_feature(cfeature.LAKES, alpha=0.5, linewidth=0.5, edgecolor="grey")

      gl = ax.gridlines(
          draw_labels=False, linewidth=0.2, color="grey", alpha=0.5, linestyle="--"
      )

      return ax

  # Plot the data and the filtered transmission lines using LineCollection
  # Scatter the devices
  def plot(tl, device_gdf, device_to_lines, extent=[-125, -66.5, 24, 50]):
      # Filter devices based on device_to_lines keys
      tl_gdf_subset_4326 = tl.to_crs("EPSG:4326")
      device_gdf_4326 = device_gdf.to_crs("EPSG:4326")

      devices = list(device_to_lines.keys())
      device_subset = device_gdf_4326[device_gdf_4326["device"].isin(devices)]

      # Gather all line IDs connected to devices
      line_ids = {line for lines in device_to_lines.values() for line in lines}
      line_subset = tl_gdf_subset_4326[tl_gdf_subset_4326["LINE_ID"].isin(line_ids)]

      # Create a list of line segments from transmission line geometries
      lines = [np.array(geom.coords) for geom in line_subset.geometry]

      # Find min and max based on the line segments
      all_coords = np.concatenate(lines, axis=0)
      min_x, min_y = np.min(all_coords, axis=0)
      max_x, max_y = np.max(all_coords, axis=0)
      bbox = Bbox.from_bounds(min_x, min_y, max_x - min_x, max_y - min_y)

      # Set up the map using cartopy
      fig = plt.figure(figsize=(12, 10))
      ax = plt.axes(projection=ccrs.PlateCarree())
      ax = setup_map(ax, spatial_extent=[min_x, max_x, min_y, max_y])

      # Add transmission lines using LineCollection
      line_collection = LineCollection(
          lines, colors="blue", linewidths=1, transform=ccrs.PlateCarree()
      )
      ax.add_collection(line_collection)

      # Scatter device locations
      ax.scatter(
          device_subset.geometry.x,
          device_subset.geometry.y,
          color="red",
          marker="o",
          s=10,
          transform=ccrs.PlateCarree(),
          zorder=3,
      )

      ax.set_extent(extent, crs=ccrs.PlateCarree())

      ax.set_title("Transmission Lines and Device Locations")
      #plt.show()

  logger.info("Adding nearset voltage to extended info_df")

  info_df['nearest_volt'] = np.nan

  # Running over all TVA and NERC GIC monitors
  if __name__ == "__main__":
      buffer_distance = 500  # specify the dist in m
      tl_gdf_subset, device_gdf, connection_dicts = process_data(
          transmission_fname, buffer_distance
      )

      (
          device_to_lines,
          line_to_device,
          device_to_line_voltages,
          device_2_device,
          line_to_device_map,
      ) = connection_dicts

      # Haversine distance usage between two devices
      # Make sure to make it earth centered earth fixed (wgs84) crs
      device_gdf = device_gdf.to_crs("EPSG:4326")
      device_1 = device_gdf.iloc[0].geometry
      device_2 = device_gdf.iloc[10].geometry
      lat1, lon1 = device_1.y, device_1.x
      lat2, lon2 = device_2.y, device_2.x
      dist = haversine_dist(lat1, lon1, lat2, lon2)
      #print(
      #    f"Distance between devices: {device_gdf.iloc[0].device} and {device_gdf.iloc[10].device}: {dist} km"
      #)

      #print(device_gdf['device'])

      # Display the resulting dictionaries
      #print("Device to Lines:")
      #print(device_to_lines)
      #print("\nLine to Device:")
      #print(line_to_device)
      #print("\nDevice to Line Voltages:")
      #print(device_to_line_voltages)
      #print("\nDevice to Device:")
      #print(device_2_device)
      #print("\nLine to Device Map:")
      #print(line_to_device_map)

      for site in device_to_line_voltages.keys():
          max_volt = max(device_to_line_voltages[site])
          if max_volt > 0:
              info_df.loc[info_df['site_id'] == site, 'nearest_volt'] = max(device_to_line_voltages[site])

      # Optional plot the data for TVA and NERC on the same plot
      plot(tl_gdf_subset, device_gdf, device_to_lines, extent=[-125, -67, 25.5, 49.5])

# Read info.csv
logger.info(f"Reading {CONFIG['files']['info']}")
info_df = pd.read_csv(CONFIG['files']['info'])

add_beta(info_df, CONFIG['files']['beta'], beta_site='OTT')
add_geomag(info_df, CONFIG['limits']['data'][0].strftime('%Y-%m-%dT%H:%M:%S'))
add_sim_site(info_df, CONFIG['files']['gmu']['sim_file'], update_csv=False)
add_voltage(info_df, CONFIG['files']['shape']['transmission_lines'])
info_df = add_power_pool(info_df, CONFIG['files']['nerc_gdf'])

# Rename 'error' column from info.csv to 'manual_error' in info.extended.csv
info_df = info_df.rename(columns={'error': 'manual_error'})

out_fname = CONFIG['files']['info_extended']
info_df.to_csv(out_fname, index=False)
logger.info(f"Wrote {out_fname}")


print(f"Preparing {CONFIG['files']['info_extended_json']}")
info_dict = infodf2dict(info_df, logger)

logger.info(f"Writing {CONFIG['files']['info_extended_json']}")
with open(CONFIG['files']['info_extended_json'], 'w') as f:
  json.dump(info_dict, f, indent=2)
