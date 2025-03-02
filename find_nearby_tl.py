# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.transforms import Bbox

# cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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


def load_devices(tv_loc, crs="EPSG:4326"):
    """
    Load device data from a CSV and convert to a GeoDataFrame.
    """
    tva_df = pd.read_csv(tv_loc, names=["device", "latitude", "longitude"])
    tva_gdf = gpd.GeoDataFrame(
        tva_df, geometry=gpd.points_from_xy(tva_df.longitude, tva_df.latitude), crs=crs
    )
    return tva_gdf


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


def process_data(data_dir: Path, buffer_distance: float = 1000):
    """
    Process transmission lines and device data.

    Parameters:
      data_dir: Path to the directory containing data.
      buffer_distance: Buffer (in meters) to apply around devices.

    Returns:
      tl_gdf_subset: GeoDataFrame of transmission lines (subset after spatial join).
      device_gdf: GeoDataFrame of devices with representative points.
      connection_dicts: A tuple of dictionaries:
          (device_to_lines, line_to_device, device_to_line_voltages, device_2_device, line_to_device_map)
    """
    # Define data locations
    tl_loc = (
        data_dir
        / "Electric__Power_Transmission_Lines"
        / "Electric__Power_Transmission_Lines.shp"
    )
    tv_loc = data_dir / "tva" / "mag" / "TVAmagmetadata.dat"

    # Load transmission lines and filter by US extent
    trans_lines_gdf = load_transmission_lines(tl_loc, crs_target="EPSG:4326")
    us_extent = {"minx": -125.0, "maxx": -66.9, "miny": 24.4, "maxy": 49.4}
    trans_lines_gdf = filter_transmission_lines(trans_lines_gdf, us_extent)

    # Explode lines for unique IDs and compute length
    trans_lines_gdf = explode_with_unique_line_id(trans_lines_gdf)
    trans_lines_gdf["length"] = trans_lines_gdf.geometry.apply(lambda x: x.length)

    # Load devices
    tva_gdf = load_devices(tv_loc, crs="EPSG:4326")

    # Reproject both GeoDataFrames to a projected CRS (NAD83)
    projected_crs = "EPSG:5070"
    tva_gdf = tva_gdf.to_crs(projected_crs)
    trans_lines_gdf = trans_lines_gdf.to_crs(projected_crs)

    # Buffer devices and create a buffered GeoDataFrame
    tva_gdf["buffered"] = tva_gdf.geometry.buffer(buffer_distance)
    buffered_gdf = gpd.GeoDataFrame(tva_gdf, geometry="buffered")

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

# Haversine distance bwetween to locs
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
    line_collection = LineCollection(lines, colors="blue", linewidths=1, transform=ccrs.PlateCarree())
    ax.add_collection(line_collection)

    # Scatter device locations
    ax.scatter(device_subset.geometry.x, device_subset.geometry.y, 
            color="red", marker="o", s=10, transform=ccrs.PlateCarree(), zorder=3)

    ax.set_title("Transmission Lines and Device Locations")
    plt.show()

# %%
if __name__ == "__main__":
    data_dir = Path("2024-AGU-data")
    buffer_distance = 1000  # specify the dist
    tl_gdf_subset, device_gdf, connection_dicts = process_data(
        data_dir, buffer_distance
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
    print(f"Distance between devices: {device_gdf.iloc[0].device} and {device_gdf.iloc[10].device}: {dist} km")  

    # Display the resulting dictionaries
    print("Device to Lines:")
    print(device_to_lines)
    print("\nLine to Device:")
    print(line_to_device)
    print("\nDevice to Line Voltages:")
    print(device_to_line_voltages)
    print("\nDevice to Device:")
    print(device_2_device)
    print("\nLine to Device Map:")
    print(line_to_device_map)

    # Optionall plot the data
    plot(tl_gdf_subset, device_gdf, device_to_lines, extent=[-90, -66.5, 24, 50])

# %%
