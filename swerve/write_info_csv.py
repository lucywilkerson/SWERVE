
# read in data sources, data types, data class from config
# then, according to that, read in site name
# add manual error for no lat/lon - maybe make this an option? (do this in add_row func)
# also manual error for no file found... hopefully not common

from requests_cache import logger


def write_info_csv():
    """
    Reads in data type, data source, data class, and event from config yaml file
    Writes info.csv file with:
        -site id
        -geographic latitude
        -geographic longitude
        -data type (B, GIC)
        -data class (measured, calculated)
        -data source (NERC, TVA, etc.)
        -event (eg. 2024-05-10)
        -manual error, can be directly added to csv later after visual inspection of data
    Saves to config_name/info.csv
    """
    import os
    import csv
    from swerve import config
    import pandas as pd

    CONFIG = config()

    logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

    data_types = CONFIG['info_kwargs']['data_type']
    data_classes = CONFIG['info_kwargs']['data_class']
    data_sources = CONFIG['info_kwargs']['data_source']
    events = CONFIG['event']

    data_dir = CONFIG['dirs']['original']

    # empty list to hold info
    info_list = []
    for event in events:
        for data_source in data_sources:
            # Check that data is available for given event and data source, if not skip to next event/data source
            event_exists = _check_event(event, data_source, data_dir, logger)
            if not event_exists:
                continue

            if data_source == 'NERC':
                # read site names for NERC measured GIC/B and save to csv
                if 'measured' in data_classes:
                    data_class = 'measured'
                    for data_type in data_types:
                        if data_type == 'GIC':
                            file_dir = os.path.join(data_dir, f'{data_source.lower()}', event, f'{data_type.lower()}')
                            file = os.path.join(file_dir, 'gic_monitors.csv')
                        elif data_type == 'B':
                            file_dir = os.path.join(data_dir, f'{data_source.lower()}', event, 'mag')
                            file = os.path.join(file_dir, 'magnetometers.csv')
                        else:
                            raise ValueError(f"Data type not recognized: {data_type}. Valid options are 'GIC' and 'B'.")
                        logger.info(f"    Reading {file}")
                        if not os.path.exists(file):
                            raise FileNotFoundError(f"File not found: {file}")
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            # skip header
                            next(rows)
                            for row in rows:
                                site_id = row[0]
                                geo_lat = float(row[1])
                                geo_lon = -float(row[2])
                                info_list = _add_info_row(info_list, site_id, geo_lat, geo_lon, data_type, data_class, data_source, event)
                # no calculated data provided by NERC, skip
                if 'calculated' in data_classes:
                    logger.info(f'   No calculated data for data source {data_source}; skipping.')

            if data_source == 'TVA':
                # read site names for TVA measured GIC/B and save to csv
                if 'measured' in data_classes:
                    data_class = 'measured'
                    for data_type in data_types:
                        # read site names for NERC measured GIC/B and save to csv
                        if data_type == 'GIC':
                            file_dir = os.path.join(data_dir, f'{data_source.lower()}', event, f'{data_type.lower()}', 'GIC-measured')
                            file = os.path.join(file_dir, 'GIC_monitors.dat')
                            if not os.path.exists(file):
                                raise FileNotFoundError(f"File not found: {file}")
                            with open(file, 'r') as csvfile:
                                rows = csv.reader(csvfile, delimiter=',')
                                # skip header
                                next(rows)
                                for row in rows:
                                    site_id = row[0]
                                    geo_lat = float(row[2])
                                    geo_lon = float(row[3])
                                    info_list = _add_info_row(info_list, site_id, geo_lat, geo_lon, data_type, data_class, data_source, event)
                        elif data_type == 'B':
                            file_dir = os.path.join(data_dir, f'{data_source.lower()}', event, 'mag')
                            file = os.path.join(file_dir, 'TVAmagmetadata.dat')
                            if not os.path.exists(file):
                                raise FileNotFoundError(f"File not found: {file}")
                            with open(file, 'r') as csvfile:
                                rows = csv.reader(csvfile, delimiter=',')
                                for row in rows:
                                    site_id = row[0]
                                    geo_lat = float(row[1])
                                    geo_lon = float(row[2])
                                    info_list = _add_info_row(info_list, site_id, geo_lat, geo_lon, data_type, data_class, data_source, event)
                        else:
                            raise ValueError(f"Data type not recognized: {data_type}. Valid options are 'GIC' and 'B'.")
                # TVA only has calculated GIC, not B. Add calculated if GIC, skip if B.                  
                if 'calculated' in data_classes:
                    data_class = 'calculated'
                    if data_type == 'GIC':
                        file_dir = os.path.join(data_dir, f'{data_source.lower()}', event, f'{data_type.lower()}', 'GIC-calculated')
                        # TODO: no nice file with listed names of calculated GIC sites for TVA!
                    else:
                        logger.info(f'   No calculated {data_type} data for data source {data_source}; skipping.')
        
            if data_source == 'Parry2025':
                # read site names for CAN data from Parry 2025 paper
                for data_type in data_types:
                    if 'measured' in data_classes:
                        data_class = 'measured'
                        if data_type == 'GIC':
                            file_dir = os.path.join(data_dir, f'{data_source.lower()}', event, f'{data_type.lower()}')
                            if not os.path.exists(file_dir):
                                logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                                continue
                            file = os.path.join(file_dir, 'parry_2025_info.csv')
                        else:
                            continue
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            # skip header
                            next(rows)
                            for row in rows:
                                site_id = row[0]
                                geo_lat = float(row[1])
                                geo_lon = float(row[2])
                                info_list = _add_info_row(info_list, site_id, geo_lat, geo_lon, data_type, data_class, data_source, event)

            if data_source == 'Parry2024':
                # read site names for CAN data from Parry 2024 paper
                for data_type in data_types:
                    if 'measured' in data_classes:
                        data_class = 'measured'
                        if data_type == 'GIC' or data_type == 'DMM':
                            file_dir = os.path.join(data_dir, f'{data_source.lower()}', event)
                            file = os.path.join(file_dir, 'parry_2024_info.csv')
                            with open(file, 'r') as csvfile:
                                rows = csv.reader(csvfile, delimiter=',')
                                # skip header
                                next(rows)
                                for row in rows:
                                    if (data_type == 'DMM' and row[3] == 'DMM') or (data_type == 'GIC' and row[3] == 'GIC'):
                                        site_id = row[0]
                                        geo_lat = float(row[1])
                                        geo_lon = float(row[2])
                                        info_list = _add_info_row(info_list, site_id, geo_lat, geo_lon, data_type, data_class, data_source, event)

            if data_source == 'Zhang2020':
                print('no info yet...')
                exit()

    # compile list into one df
    info_df = pd.DataFrame(info_list)
    print(info_df)
    info_fname = CONFIG['files']['info']
    logger.info(f'   Saving info to {info_fname}')
    info_df.to_csv(info_fname, index=False)
    return

def _add_info_row(info_list, site_id, geo_lat, geo_lon, data_type, data_class, data_source, event, manual_error=None):
    # adds all site information to list of sites
    if manual_error is None:
        manual_error = ''
    info_list.append({'site_id': site_id,
                    'geo_lat': geo_lat,
                    'geo_lon': geo_lon,
                    'data_type': data_type,
                    'data_class': data_class,
                    'data_source': data_source,
                    'event': event,
                    'manual_error': manual_error})
    return info_list

def _check_event(event,data_source,data_dir,logger):
    # checks if events are valid for all data sources, if not logger warning
    import os
    event_folder = os.path.join(data_dir, f'{data_source.lower()}', event)
    if not os.path.exists(event_folder):
        logger.warning(f"   Event {event} not found for data source {data_source}. Skipping...")
        return False
    else:
        logger.info(f"   Adding sites for event {event} from data source {data_source} to info.csv.")
        return True
