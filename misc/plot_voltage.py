
from swerve import config, savefig

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

figsize = (12, 6)

def read_lines(shp_file):
    import geopandas as gpd
    # US Transmission lines
    logger.info(f"Reading {shp_file}")
    trans_lines_gdf = gpd.read_file(shp_file)
    trans_lines_gdf.rename({"ID":"line_id"}, inplace=True, axis=1)
    trans_lines_gdf = trans_lines_gdf.to_crs("EPSG:4326")
    # Translate MultiLineString to LineString geometries, taking only the first LineString
    arg = trans_lines_gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
    trans_lines_gdf.loc[arg] = trans_lines_gdf.loc[arg].apply(lambda x: list(x.geoms)[0])
    # Get rid of erroneous 1 MV and low-power line voltages
    # trans_lines_gdf = trans_lines_gdf[(trans_lines_gdf["VOLTAGE"] >= 200)]
    voltages = trans_lines_gdf["VOLTAGE"].unique()
    # Order voltages from lowest to highest
    voltages = sorted(voltages)
    return voltages, trans_lines_gdf

def plot_counts():
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.scatter(plot_volts, voltage_counts, color='k', zorder=2)
    for voltage, count in zip(plot_volts, voltage_counts):
        plt.vlines(voltage, 0, count, color='k')
    plt.xlabel('Voltage [kV]')
    plt.ylabel('Number of Lines')
    plt.title(f'US Transmission Lines from HIFLD: {len(gdf)}')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, zorder=1)
    savefig('_results\plot_voltage', 'trans_lines_count', logger)
    plt.close()

def plot_lengths():
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.scatter(plot_volts, voltage_lengths, color='k', zorder=2)
    for voltage, length in zip(plot_volts, voltage_lengths):
        plt.vlines(voltage, 0, length, color='k')
    plt.xlabel('Voltage [kV]')
    plt.ylabel('Total Length [km]')
    plt.title('Length of US Transmission Lines by Voltage')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, zorder=1)
    savefig('_results\plot_voltage', 'trans_lines_length', logger)
    plt.close()

voltages, gdf = read_lines(CONFIG['files']['shape']['transmission_lines'])

plot_volts = []
voltage_counts = []
voltage_lengths = []
gdf['length_km'] = gdf['geometry'].length * (111.32)
for voltage in voltages:
    if voltage < 0:
        continue
    voltage_gdf = gdf[(gdf["VOLTAGE"] == voltage)]
    n_lines = len(voltage_gdf)
    total_length = voltage_gdf['length_km'].sum()
    logger.info(f"Voltage: {voltage} kV, # Lines: {n_lines}; Total Length: {total_length:.2f} km")
    plot_volts.append(voltage)
    voltage_counts.append(n_lines)
    voltage_lengths.append(total_length)

plot_counts()
plot_lengths()
