import math
import pandas as pd
from swerve import cli, config, read_info_df, site_read

args = cli('nearest_mag.py') 
CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
event = args['event']
if event is None:
    event = '2024-05-10'
    logger.info(f"No event specified, defaulting to '{event}'.")

def nearest_mag_sites(site_id, info_df):
    """
    Find the nearest magnetic (B) measurement sites to a given GIC site.
    
    Parameters:
    -----------
    site_id : str
        The site ID of the GIC measurement site
    info_df : pd.DataFrame
        DataFrame containing site information with columns: site_id, data_type, 
        data_class, geo_lat, geo_lon
    
    Returns:
    --------
    pd.DataFrame
        B sites sorted by haversine distance (closest to farthest)
    """
    
    # Find the GIC site
    gic_site = info_df[(info_df['site_id'] == site_id) & 
                        (info_df['data_type'] == 'GIC') & 
                        (info_df['data_class'] == 'measured')]
    
    if gic_site.empty:
        raise ValueError(f"Site {site_id} not found with data_type='GIC' and data_class='measured'")
    
    gic_lat = gic_site['geo_lat'].values[0]
    gic_lon = gic_site['geo_lon'].values[0]
    
    # Get all B measurement sites
    b_sites = info_df[(info_df['data_type'] == 'B') & 
                      (info_df['data_class'] == 'measured') &
                      (info_df['data_source'] != 'TEST')].copy()
    
    # Calculate haversine distance
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    b_sites['distance'] = b_sites.apply(
        lambda row: haversine(gic_lat, gic_lon, row['geo_lat'], row['geo_lon']), 
        axis=1
    )
    
    return b_sites.sort_values('distance').reset_index(drop=True)

info_df = read_info_df(exclude_errors=True)
gic_df = info_df[(info_df['data_type'] == 'GIC') & (info_df['data_class'] == 'measured') 
                 & (info_df['data_source'] != 'TEST')].copy()

nearest_b_df = pd.DataFrame()
for sid in gic_df['site_id'].unique():
    nearest_b_sites = nearest_mag_sites(sid, info_df)
    gic_site = gic_df[gic_df['site_id'] == sid]
    nearest_b_df = pd.concat([
        nearest_b_df,
        pd.DataFrame({
            'gic_sid': [sid],
            'gic_lat': [gic_site['geo_lat'].values[0]],
            'gic_lon': [gic_site['geo_lon'].values[0]],
            'nearest_b_sid': [nearest_b_sites['site_id'].iloc[0]],
            'b_lat': [nearest_b_sites['geo_lat'].iloc[0]],
            'b_lon': [nearest_b_sites['geo_lon'].iloc[0]],
            'distance_km': [nearest_b_sites['distance'].iloc[0]]
        })
    ], ignore_index=True)

logger.info(f"  Saving nearest B sites for GIC sites to info/{event}/nearest_b_sites.csv")
nearest_b_df.to_csv(f"info/{event}/nearest_b_sites.csv", index=False)


exit()


#######################################################################

gic_sids = ['Bull Run', 'Montgomery', 'Union', 'Widows Creek']

for sid in gic_sids:
    nearest_b_sites = nearest_mag_sites(sid, info_df)
    print(f"Nearest B sites to {sid}:\n{nearest_b_sites[['site_id', 'distance']].head()}\n")

    data = {}
    # Read data from GIC and nearest B site
    gic_data = site_read(sid, data_types='GIC', logger=logger)['GIC']['measured']
    b_data = site_read(nearest_b_sites['site_id'].iloc[0], data_types='B', logger=logger)['B']['measured']
    
    # Extract modified data
    gic = next(iter(gic_data.values()))['modified']
    b = next(iter(b_data.values()))['modified']
    
    # Align timestamps
    common_times = pd.Index(gic['time']).intersection(pd.Index(b['time']))
    
    # Create combined dataframe
    site_df = pd.DataFrame({
        'datetime': common_times,
        'gic': pd.Series(gic['data'].flatten(), index=gic['time']).reindex(common_times).values,
        'bx': pd.Series(b['data'][:, 0], index=b['time']).reindex(common_times).values,
        'by': pd.Series(b['data'][:, 1], index=b['time']).reindex(common_times).values,
        'bz': pd.Series(b['data'][:, 2], index=b['time']).reindex(common_times).values
    })
    print(site_df)
    # save as CSV
    site_df.to_csv(f"../timeseries-predict/data/raw/{sid.lower().replace(' ', '')}/{sid.lower().replace(' ', '')}_gic_b.csv", index=False)
    exit()
