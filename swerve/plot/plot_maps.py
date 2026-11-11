import os

import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.interpolate import LinearNDInterpolator

from swerve import config, savefig

CONFIG = config()
if 'files' not in CONFIG:
    raise FileNotFoundError(
        "No info file available because no --config was specified. "
        "Run info.py --config <config.yaml> for the config before running "
        "plot_maps with the same --config."
    )
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

paper = False # Saves map to paper dir
out_dir = os.path.join(CONFIG['dirs']['data'], 'data_processed', 'summary', 'map')
if paper:
    paper_dir = '../SWERVE_data_paper'

auto_crop = False # Crop each location map to its sites if True
state = True # Show political boundaries if True

crs = ccrs.PlateCarree()
transform = ccrs.PlateCarree()
projection = ccrs.Miller()

def add_features(ax, state):
    # Add coastlines and other features
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    lakes = cfeature.NaturalEarthFeature('physical', 'lakes', '50m',
                                         edgecolor='face',
                                         facecolor='lightblue')
    #ax.add_feature(lakes)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='darkgray')
    if state:
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='darkgray')

def add_symbols(ax, df, transform, markersize, add_key=False):

    default_symbols = {
            ('DMM', 'measured'): ('o', 'r', 'r'),
            ('GIC', 'measured'): ('x', 'c', 'c'),
            ('B', 'measured'): ('^', 'k', 'k'),
            ('DMM', 'calculated'): ('^', 'r', 'none'),
            ('GIC', 'calculated'): ('^', 'k', 'none'),
            ('B', 'calculated'): ('^', 'b', 'none')
        }

    plotted_symbols = {}
    for row in df.itertuples():
        # Select the symbol
        data_key = (row.data_type, row.data_class)
        symbol = default_symbols.get(data_key)

        # Add specific color for INTERMAG
        if (getattr(row, 'data_source', None), row.data_class) == ('INTERMAG', 'measured'):
            symbol = ('o', 'k', 'lightgray')

        if symbol is None:
            continue

        marker, color, face = symbol
        legend_label = ('INTERMAG'
                        if (getattr(row, 'data_source', None), row.data_class)
                        == ('INTERMAG', 'measured')
                        else f'{row.data_type} {row.data_class}')
        plotted_symbols.setdefault(legend_label, symbol)

        ax.plot(row.geo_lon, row.geo_lat,
                mfc=face, marker=marker, color=color,
                markersize=markersize,
                transform=transform)

    if add_key:
        import matplotlib.lines as mlines
        legend_elements = [
            mlines.Line2D([], [], color=color, marker=marker,
                          markerfacecolor=face, markeredgecolor=color,
                          linestyle='None', label=label)
            for label, (marker, color, face) in plotted_symbols.items()
        ]
        ax.legend(handles=legend_elements, loc='lower left')

def _auto_crop(info_df):
  if info_df.empty:
    return [-180, 180, -90, 90]
  coords = info_df[['geo_lon', 'geo_lat']].to_numpy(dtype=float)
  if (not np.isfinite(coords).all()
      or (np.abs(coords[:, 0]) > 180).any()
      or (np.abs(coords[:, 1]) > 90).any()):
    raise ValueError("Cannot fit map: sites must have valid geographic coordinates.")
  lon_min, lat_min = coords.min(axis=0)
  lon_max, lat_max = coords.max(axis=0)
  lon_pad = max((lon_max - lon_min) * 0.05, 2)
  lat_pad = max((lat_max - lat_min) * 0.05, 2)
  return [max(-180, lon_min - lon_pad), min(180, lon_max + lon_pad),
          max(-90, lat_min - lat_pad), min(90, lat_max + lat_pad)]

def location_map(info_df, extent, markersize, out_dir, out_name, add_key=False, mag_lat=False, patch=False,
                 date='2024-05-11T00:00:00'):
  if info_df.empty:
    extent = [-180, 180, -90, 90]
    logger.warning(f"Map '{out_name}' will be empty because no sites are available; using full-globe extent.")
  elif extent is None:
    extent = _auto_crop(info_df)
  # Create a figure and axes with a specific projection
  fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})
  # Adding map features and symbols for locations
  add_features(ax, state=False)
  if patch:
    patch_kwargs = {"fc": 'lightyellow', "ec": 'g', "transform": transform}
    ax.add_patch(patches.Rectangle([-91, 33], 9, 5, **patch_kwargs))

  if mag_lat:
        from spacepy import coordinates as coord
        from spacepy.time import Ticktock

        # Create a grid of geographic latitude and longitude for CONUS
        # Using a 1-degree resolution for a balance of detail and performance
        lat_res, lon_res = 1, 1
        lat_min, lat_max = extent[2], extent[3]
        lon_min, lon_max = extent[0], extent[1]

        lats_geo = np.arange(lat_min, lat_max + lat_res, lat_res)
        lons_geo = np.arange(lon_min, lon_max + lon_res, lon_res)
        lons_geo_grid, lats_geo_grid = np.meshgrid(lons_geo, lats_geo)

        # Reshape the grids into a list of coordinate pairs (altitude, latitude, longitude)
        altitude_re = np.ones_like(lats_geo_grid.ravel())  # 1 Earth radius above center (surface)
        coords_geo = np.vstack([altitude_re, lats_geo_grid.ravel(), lons_geo_grid.ravel()]).T

        # Convert geographic to geomagnetic coordinates
        geo_c = coord.Coords(coords_geo, 'GEO', 'sph', units=['Re', 'deg', 'deg'])
        geo_c.ticks = Ticktock([date] * coords_geo.shape[0], 'UTC')

        # Convert the coordinates to the geomagnetic (MAG) system
        mag_c = geo_c.convert('MAG', 'sph')

        # Extract the geomagnetic latitude and longitude, and reshape to the original grid
        mlats = mag_c.lati.reshape(lats_geo_grid.shape)

        # Plot the data using cartopy
        lat_contour = ax.contour(lons_geo_grid, lats_geo_grid, mlats,
                                transform=ccrs.PlateCarree(),
                                colors='gray', linestyles='dashed',
                                linewidths=0.5,
                                levels=np.arange(30, 60, 2))
        # Generate label locations along the right edge for each contour level
        lat_edge = extent[1]-2
        contour_levels = np.arange(int(extent[2]+2), int(extent[3]+2), 2)
        label_loc_geo = [(lat_edge, level) for level in contour_levels]
        label_loc = [projection.transform_point(lon, lat, ccrs.PlateCarree()) for lon, lat in label_loc_geo]
        ax.clabel(lat_contour, inline=True, manual=label_loc, fontsize=10, fmt='%.1f°')


  add_symbols(ax, info_df, transform, markersize, add_key=add_key)
  # Set the extent of the map
  if list(extent) == [-180, 180, -90, 90]:
    ax.set_global()
  else:
    ax.set_extent(extent, crs=crs)
  # Save map
  savefig(out_dir, out_name, logger)

def cc_vs_dist_map(info_df, cc_df):

    def add_pair_lines(ax, site_1_id, site_2_id, color='k'):
        # function to draw lines between pairs
        # TODO: merge with map_cc
        site_1_lat = info_df.loc[info_df['site_id'] == site_1_id, 'geo_lat'].values[0]
        site_1_lon = info_df.loc[info_df['site_id'] == site_1_id, 'geo_lon'].values[0]

        site_2_lat = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lat'].values[0]
        site_2_lon = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lon'].values[0]

        ax.plot([site_1_lon, site_2_lon], [site_1_lat, site_2_lat],
                color=color, transform=transform)

    # mapping top and bottom site pairs
    cc_lims = [0.8,0.6,0.4,0.2,0.01]
    # distance limit for cc < 0.4
    dist_lim = 1000

    #setting up maps
    fig, axs = plt.subplots(2,3, figsize=(13, 5), subplot_kw={'projection': projection})
    for ax in axs.flat:
        add_features(ax, state)
        ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)

    #values to count number of pairs
    num_pairs = np.zeros(6)
    # looping over pairs
    for idx, row in cc_df.iterrows():
        cc = np.abs(row['cc'])
        site_1_id = row['site_1']
        site_2_id = row['site_2']
        if cc > cc_lims[0]:
            add_pair_lines(axs[0,0], site_1_id, site_2_id, color='r')
            num_pairs[0] += 1
        elif cc > cc_lims[1]:
            add_pair_lines(axs[0,1], site_1_id, site_2_id)
            num_pairs[1] += 1
        elif cc > cc_lims[2]:
            add_pair_lines(axs[0,2], site_1_id, site_2_id)
            num_pairs[2] += 1
        elif cc > cc_lims[3]:
            if row['dist(km)'] > dist_lim:
                continue
            add_pair_lines(axs[1,0], site_1_id, site_2_id)
            num_pairs[3] += 1
        elif cc > cc_lims[4]:
            if row['dist(km)'] > dist_lim:
                continue
            add_pair_lines(axs[1,1], site_1_id, site_2_id)
            num_pairs[4] += 1
        elif cc < cc_lims[4]:
            if row['dist(km)'] > dist_lim:
                continue
            add_pair_lines(axs[1,2], site_1_id, site_2_id)
            num_pairs[5] += 1

    labels = [
        f"{int(num_pairs[0])} sites with |cc| > {cc_lims[0]}",
        f"{int(num_pairs[1])} sites with {cc_lims[0]} > |cc| > {cc_lims[1]}",
        f"{int(num_pairs[2])} sites with {cc_lims[1]} > |cc| > {cc_lims[2]}",
        f"{int(num_pairs[3])} sites with {cc_lims[2]} > |cc| > {cc_lims[3]}",
        f"{int(num_pairs[4])} sites with {cc_lims[3]} > |cc| > {cc_lims[4]}",
        f"{int(num_pairs[5])} sites with |cc| < {cc_lims[4]}"
    ]

    for ax, label in zip(axs.flat, labels):
        ax.text(0.5, -0.1, label, ha="center", transform=ax.transAxes, fontsize=8)
    plt.tight_layout()
    savefig(out_dir, 'cc_vs_dist_map', logger)
    plt.close()

def std_map(info_df, cc_df):
    # plotting standard deviation
    # Create a figure and axes with a specific projection
    fig, ax = plt.subplots(1, figsize=(10, 8), subplot_kw={'projection': projection})
    # setting up map
    add_features(ax, state)
    # Set the extent of the map (USA)
    ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)
    plt.title('Standard deviation of GIC data')
    # plotting map w lines
    for idx_1, row_1 in info_df.iterrows():
        site_id = row_1['site_id']
        site_lat = info_df.loc[info_df['site_id'] == site_id, 'geo_lat'].values[0]
        site_lon = info_df.loc[info_df['site_id'] == site_id, 'geo_lon'].values[0]
        for idx_2, row_2 in cc_df.iterrows():
            if row_2['site_1'] == site_id:
                std = row_2['std_1']
            else:
                continue
            ax.plot(site_lon, site_lat, color='k', marker='o', markersize=std, transform=transform)
    savefig(out_dir, 'std_map', logger)
    plt.close()

def site_maps(info_df, cc_df):

    def map_cc(ax, site_id, cc_df, colors=False, lines=False):

        def add_cc_colors(cc):
            if cc < 0.2:
                color = 'red'
                mark = 2
            elif cc < 0.4:
                color = 'orange'
                mark = 2.75
            elif cc < 0.6:
                color = 'yellow'
                mark = 3.5
            elif cc < 0.8:
                color = 'lightgreen'
                mark = 4.25
            else:
                color = 'green'
                mark = 5
            return color, mark

        def add_cc_width(cc):

            if cc < 0.2:
                width = 1
            elif cc < 0.4:
                width = 2
            elif cc < 0.6:
                width = 3
            elif cc < 0.8:
                width = 4
            else:
                width = 5
            return width

        site_1_lat = info_df.loc[info_df['site_id'] == site_id, 'geo_lat'].values[0]
        site_1_lon = info_df.loc[info_df['site_id'] == site_id, 'geo_lon'].values[0]

        for idx, row in cc_df.iterrows():
            if row['site_1'] == site_id:
                site_2_id = row['site_2']
            elif row['site_2'] == site_id:
                site_2_id = row['site_1']
            else:
                continue
            cc = np.abs(row['cc'])
            site_2_lat = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lat'].values[0]
            site_2_lon = info_df.loc[info_df['site_id'] == site_2_id, 'geo_lon'].values[0]
            if colors == True:
                col, marksiz= add_cc_colors(cc)
                ax.plot(site_2_lon, site_2_lat, color=col, 
                        marker='o', markersize=marksiz, transform=transform)
                prim_color='k'
            elif lines == True:
                width = add_cc_width(cc)
                ax.plot([site_1_lon, site_2_lon], [site_1_lat, site_2_lat],
                        color='k', linewidth=width, transform=transform)
                prim_color='r'
            else:
                ax.plot(site_2_lon, site_2_lat, 'bo', transform=transform)
                prim_color='k'
            ax.plot(site_1_lon, site_1_lat, color=prim_color, marker='*', transform=transform)

    # Plotting maps and cc plots for each site
    for idx_1, row in info_df.iterrows():
        site_1_id = row['site_id']

        # Create a figure and axes with a specific projection
        _, axs = plt.subplots(2, figsize=(10, 8), subplot_kw={'projection': projection})
        # setting up map
        for ax in axs.flat:
            add_features(ax, state)
            # Set the extent of the map (USA)
            ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)

        # plotting map w lines
        map_cc(axs[0], site_1_id, cc_df, lines=True)
        # plotting map w colors
        map_cc(axs[1], site_1_id, cc_df, colors=True)
        plt.suptitle(site_1_id)
        # saving figure
        sid = site_1_id
        fdir = os.path.join('data_processed', 'sites', sid.lower().replace(' ', ''))
        savefig(fdir, 'cc_vs_dist_map', logger)
        plt.close()

def beta_maps():

    def plot(df, beta_site):
        # Create the interpolator
        interpolator = LinearNDInterpolator(df[['lat', 'lon']], df['beta'])

        # Define a grid for interpolation
        lat_grid = np.linspace(df['lat'].min(), df['lat'].max(), 100)
        lon_grid = np.linspace(df['lon'].min(), df['lon'].max(), 100)
        lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)

        # Interpolate beta values onto the grid
        beta_grid = interpolator(lat_grid, lon_grid)

        # Plot the interpolated data on a map
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max()], crs=ccrs.PlateCarree())

        # Add features to the map
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        # Plot the interpolated beta values
        contour = ax.contourf(lon_grid, lat_grid, beta_grid, transform=ccrs.PlateCarree(), cmap='viridis')

        cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.05, aspect=50)
        cbar.set_label('Interpolated Beta')
        ax.set_title(beta_site)

        savefig(out_dir, f'beta_map_{beta_site.lower()}', logger)
        plt.close()

    from scipy.io import loadmat

    data = loadmat(CONFIG['files']['beta'])
    data = data['waveform'][0]

    # Convert the MATLAB data to a pandas DataFrame
    raw_df = pd.DataFrame(data)

    beta_sites = ['MEA', 'OTT', 'MMB', 'NUR']

    for idx, beta_site in enumerate(beta_sites):
        betas = raw_df[0][idx][0][0][1][0]

        # Convert betas list to a Pandas DataFrame
        rows = []
        for i in range(len(betas)):
            beta = betas[i][0][0][1][0][0]
            lat = betas[i][0][0][2][0][0]
            lon = betas[i][0][0][3][0][0]
            rows.append([beta, lat, lon])

        df = pd.DataFrame(rows, columns=['beta', 'lat', 'lon'])
        plot(df, beta_site)


def transmission_map(info_df, gdf, cc_df, std=False):

    def add_trans_lines(ax, gdf):
        gdf = gdf.to_crs("EPSG:4326")
        # Translate MultiLineString to LineString geometries, taking only the first LineString
        gdf.loc[
        gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
        ] = gdf.loc[
        gdf["geometry"].apply(lambda x: x.geom_type) == "MultiLineString", "geometry"
        ].apply(lambda x: list(x.geoms)[0])
        # Get rid of erroneous 1MV and low power line voltages
        gdf = gdf[(gdf["VOLTAGE"] >= 200)]
        # Plot the lines
        for idx, row in gdf.iterrows():
            x, y = row['geometry'].xy
            ax.plot(x, y, color='black', linewidth=1,transform=transform)

    def add_loc(ax, info_df, cc_df, stdev=False):
        # plotting standard deviation
        for idx_1, row_1 in info_df.iterrows():
            site_id = row_1['site_id']
            site_lat = info_df.loc[info_df['site_id'] == site_id, 'geo_lat'].values[0]
            site_lon = info_df.loc[info_df['site_id'] == site_id, 'geo_lon'].values[0]
            if stdev == True:
                for idx_2, row_2 in cc_df.iterrows():
                    if row_2['site_1'] == site_id: #TODO: with new cc.pkl format, this doesn't work
                        std = row_2['std_1']
                    else:
                        continue
                    ax.plot(site_lon, site_lat, color='r', marker='o', markersize=std, transform=transform)
            else:
                ax.plot(site_lon, site_lat, color='r', marker='o', markersize=5, transform=transform)

    # Setting up map
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})
    add_features(ax, state)
    # Set the extent of the map (USA)
    ax.set_extent([-125, -67, 25.5, 49.5], crs=crs)
    # Add transmission lines
    add_trans_lines(ax, gdf)
    # Add GIC sites
    add_loc(ax, info_df, cc_df, stdev=std)
    if std:
        ax.set_title(r'US Transmission Lines $\geq$ 200kV w GIC Standard Deviation', fontsize=15)
        savefig(out_dir, 'transmission_std_map', logger)
    else:
        ax.set_title(r'US Transmission Lines $\geq$ 200kV w "good" GIC Sites', fontsize=15)
        savefig(out_dir, 'transmission_map', logger)


def _add_intermag(class_df):
    intermag_info_file = os.path.join(
        CONFIG['dirs']['original'], 'intermagnet', 'intermagnet_info.csv'
    )
    if not os.path.exists(intermag_info_file):
        raise FileNotFoundError(f"File not found: {intermag_info_file}")
    intermag_info = pd.read_csv(intermag_info_file, skiprows=1)
    for site_id, geo_lat, geo_lon in zip(
            intermag_info['IAGA code'], intermag_info['Latitude'], intermag_info['Longitude']):
        class_df = pd.concat([class_df, pd.DataFrame([{
            'site_id': site_id,
            'geo_lat': float(geo_lat),
            'geo_lon': float(geo_lon-360) if float(geo_lon) > 180 else float(geo_lon),
            'data_source': 'INTERMAG',
            'data_type': 'B',
            'data_class': 'measured',
            'event': list(CONFIG['event'].keys())[0]
        }])], ignore_index=True)
    return class_df


if __name__ == "__main__":
    info_file = CONFIG['files']['info']
    if not os.path.exists(info_file):
        raise FileNotFoundError(
            f"Site metadata '{info_file}' not found. Run info.py with the same "
            "--config before running plot_maps."
        )
    logger.info(f"Reading {info_file}")
    info_df = pd.read_csv(info_file, dtype={'site_id': str, 'event': str})
    for data_class in ['measured', 'calculated']:
        class_df = info_df[info_df['data_class'] == data_class]
        class_df = class_df.drop_duplicates(
            subset=['site_id', 'geo_lat', 'geo_lon', 'data_source', 'data_type']
        )
        # Add INTERMAG sites if INTERMAG not in info_df but in config data_sources
        if data_class == 'measured' and 'INTERMAG' not in class_df['data_source'].unique() and (CONFIG['info_kwargs']['data_source'] is None or 'INTERMAG' in CONFIG['info_kwargs']['data_source']):
            class_df = _add_intermag(class_df)
        # Draw B first, then DMM, with GIC on top
        class_df = class_df.sort_values(
            'data_type',
            key=lambda data_type: data_type.map({'B': 0, 'DMM': 1, 'GIC': 2}),
            kind='stable'
        )
        logger.info(f"Number of unique sites for {data_class}: {class_df['site_id'].nunique()}")
        # Make one map for each data class, including all data types.
        extent = None if auto_crop else [-180, 180, -80, 85]
        location_map(class_df, extent, 2, out_dir, f'{data_class}_locations', add_key=True)
        plt.close()
    # If paper, combine all measured plots into stack plot with a), b), c) labels
    if paper:
        from PIL import Image, ImageDraw, ImageFont
        from matplotlib import font_manager

        images = []
        for dt in [['GIC', 'DMM'], 'B']:
            if isinstance(dt, list):
                dt_str = "_".join([d.lower() for d in dt])
            else:
                dt_str = dt.lower()
            with Image.open(os.path.join(out_dir, f'measured_{dt_str}_locations.png')) as image:
                images.append(image.convert('RGB'))
        widths, heights = zip(*(i.size for i in images))
        total_height = sum(heights)
        max_width = max(widths)
        combined = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for idx, im in enumerate(images):
            combined.paste(im, (0, y_offset))
            y_offset += im.size[1]
        # Add labels a), b), c), etc. to each stacked image
        draw = ImageDraw.Draw(combined)
        font_size = max(24, int(max_width * 0.03))
        font = ImageFont.truetype(
            font_manager.findfont(font_manager.FontProperties(weight='bold')),
            size=font_size
        )
        margin = max(10, int(max_width * 0.02))
        y_offset = 0
        for idx, im in enumerate(images):
            label = chr(97 + idx) + ')'  # a), b), c)
            draw.text((margin, y_offset + margin), label, fill='black', font=font,
                      anchor='lt', stroke_width=max(1, font_size // 30), stroke_fill='white')
            y_offset += im.size[1]
        os.makedirs(paper_dir, exist_ok=True)
        for file_type in ['.pdf', '.png']:
            stack_file = os.path.join(paper_dir, f'measured_locations_stack{file_type}')
            logger.info(f"Writing {stack_file}")
            combined.save(stack_file)
