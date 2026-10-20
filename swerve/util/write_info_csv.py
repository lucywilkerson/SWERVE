import csv
import datetime
import os
import re


_DATA_RANGE_CACHE = {} # To save the date range of each annual data file to avoid re-reading


def write_info_csv():
    """Write the configured site metadata to info.csv."""
    import pandas as pd
    from swerve import config, read_intermag_info
    from .source_catalog import ANNUAL_SOURCES, source_partitions

    def _add_info_row(info_list, site_id, geo_lat, geo_lon, data_type, data_class, data_source, event, manual_error=None):
        if manual_error is None:
            manual_error = ''
        info_list.append({
            'site_id': site_id,
            'geo_lat': geo_lat,
            'geo_lon': geo_lon,
            'data_type': data_type,
            'data_class': data_class,
            'data_source': data_source,
            'event': event,
            'manual_error': manual_error,
        })
        return info_list

    CONFIG = config()
    logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
    data_types = CONFIG['info_kwargs']['data_type']
    data_classes = CONFIG['info_kwargs']['data_class']
    data_sources = CONFIG['info_kwargs']['data_source']
    events = CONFIG['event']
    data_dir = CONFIG['dirs']['original']

    info_list = []
    for event in events:
        # Checking data availability for each source and event combination
        for data_source in data_sources:
            if not _source_available(data_source, data_dir, logger):
                continue
            if events[event].get('event_type') == 'annual':
                if data_source not in ANNUAL_SOURCES:
                    logger.info(f"   Skipping non-annual source {data_source} for annual event {event}.")
                    continue
                if str(event) not in source_partitions(data_dir, data_source):
                    logger.info(f"   Annual source {data_source} has no data for {event}. Skipping...")
                    continue
            if (data_source not in ANNUAL_SOURCES and data_source != 'INTERMAG'
                    and not _event_available(event, data_source, data_dir, logger)):
                continue
            if data_source in ANNUAL_SOURCES:
                logger.info(f"   Checking annual data from source {data_source} for event {event}.")
            if data_source == 'INTERMAG':
                if 'measured' in data_classes and 'B' in data_types:
                    data_class = 'measured'
                    data_type = 'B'
                    start, stop = CONFIG['event'][event]['data_limits']
                    source_sites = read_intermag_info(start, stop, logger=logger)
                    for site_id, geo_lat, geo_lon in zip(
                            source_sites['sites'], source_sites['lat'], source_sites['lon']):
                        info_list = _add_info_row(
                            info_list, site_id, geo_lat, geo_lon,
                            data_type, data_class, data_source, event)
                else:
                    logger.info(f'   Only measured B data for data source {data_source}; skipping.')

            elif data_source == 'NERC':
                if 'measured' in data_classes:
                    data_class = 'measured'
                    for data_type in data_types:
                        if data_type == 'GIC':
                            file_dir = os.path.join(data_dir, data_source.lower(), event, data_type.lower())
                            file = os.path.join(file_dir, 'gic_monitors.csv')
                        elif data_type == 'B':
                            file_dir = os.path.join(data_dir, data_source.lower(), event, 'mag')
                            file = os.path.join(file_dir, 'magnetometers.csv')
                        else:
                            raise ValueError(f"Data type not recognized: {data_type}. Valid options are 'GIC' and 'B'.")
                        logger.info(f"    Reading {file}")
                        if not os.path.exists(file):
                            raise FileNotFoundError(f"File not found: {file}")
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            next(rows)
                            for row in rows:
                                info_list = _add_info_row(info_list, row[0], float(row[1]), -float(row[2]), data_type, data_class, data_source, event)
                if 'calculated' in data_classes:
                    logger.info(f'   No calculated data for data source {data_source}; skipping.')

            elif data_source == 'TVA':
                if 'measured' in data_classes:
                    data_class = 'measured'
                    for data_type in data_types:
                        if data_type == 'GIC':
                            file_dir = os.path.join(data_dir, data_source.lower())
                            file = os.path.join(file_dir, 'GIC_monitors.dat')
                            if not os.path.exists(file):
                                raise FileNotFoundError(f"File not found: {file}")
                            event_dir = os.path.join(file_dir, event, f'{data_type.lower()}', 'GIC-measured')
                            # Getting event sites from file names
                            event_sites = [os.path.splitext(f)[0].split('_', 1)[0].removeprefix('gic-') for f in os.listdir(event_dir) if os.path.isfile(os.path.join(event_dir, f))]
                            with open(file, 'r') as csvfile:
                                rows = csv.reader(csvfile, delimiter=',')
                                next(rows)
                                for row in rows:
                                    # Deal with special case for Widows Creek 1 (originally just Widows Creek until second monitor was added)
                                    if row[0] == 'Widows Creek 1' and 'widowscreek' in event_sites:
                                        row[0] = 'Widows Creek'
                                    # Handle special case for Paradise sites during the 2024-05-10 event
                                    if event == '2024-05-10' and row[0] == 'Paradise':
                                        for paradise_site in ('Paradise 2', 'Paradise 3'):
                                            if paradise_site.lower().replace(' ', '') in event_sites:
                                                info_list = _add_info_row(info_list, paradise_site, float(row[2]), float(row[3]), data_type, data_class, data_source, event)
                                        continue
                                    # Check for what events the passed site has
                                    if row[0].lower().replace(' ', '') not in event_sites:
                                        print(f"   Site {row[0]} not found in event sites for event {event}; skipping.")
                                        continue
                                    info_list = _add_info_row(info_list, row[0], float(row[2]), float(row[3]), data_type, data_class, data_source, event)
                        elif data_type == 'B':
                            file_dir = os.path.join(data_dir, data_source.lower(), event, 'mag')
                            if not os.path.exists(file_dir):
                                logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                                continue
                            file = os.path.join(file_dir, 'TVAmagmetadata.dat')
                            if not os.path.exists(file):
                                raise FileNotFoundError(f"File not found: {file}")
                            # Getting event sites from file names
                            event_sites = [os.path.splitext(f)[0].split('_', 1)[0] for f in os.listdir(file_dir) if (os.path.isfile(os.path.join(file_dir, f)) and f.endswith('.csv'))]
                            with open(file, 'r') as csvfile:
                                rows = csv.reader(csvfile, delimiter=',')
                                for row in rows:
                                    if row[0].lower().replace(' ', '') not in event_sites:
                                        print(f"   Site {row[0]} not found in event sites for event {event}; skipping.")
                                        continue
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)
                        else:
                            logger.warning(f'   Data type {data_type} not available for data source {data_source}. Skipping...')
                if 'calculated' in data_classes:
                    data_class = 'calculated'
                    if 'GIC' not in data_types:
                        logger.info(f'   No calculated B data for data source {data_source}; skipping.')

            elif data_source == 'Parry2025':
                for data_type in data_types:
                    if 'measured' in data_classes:
                        data_class = 'measured'
                        if data_type == 'GIC':
                            file_dir = os.path.join(data_dir, data_source.lower(), event, data_type.lower())
                        elif data_type == 'B':
                            file_dir = os.path.join(data_dir, data_source.lower(), event, 'mag')
                        else:
                            continue
                        if not os.path.exists(file_dir):
                            logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                            continue
                        file = os.path.join(file_dir, 'parry_2025_info.csv')
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            next(rows)
                            for row in rows:
                                info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)

            elif data_source == 'Parry2024':
                for data_type in data_types:
                    if 'measured' in data_classes:
                        data_class = 'measured'
                        if data_type in ['GIC', 'DMM']:
                            file = os.path.join(data_dir, data_source.lower(), event, 'parry_2024_info.csv')
                            with open(file, 'r') as csvfile:
                                rows = csv.reader(csvfile, delimiter=',')
                                next(rows)
                                for row in rows:
                                    if row[3] == data_type:
                                        info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)

            elif data_source in ['Marsal2025', 'Marsal2021']:
                for data_type in data_types:
                    if data_type == 'DMM' and 'measured' in data_classes:
                        data_class = 'measured'
                        event_dir = os.path.join(data_dir, data_source.lower(), event, data_type.lower())
                        if os.path.isdir(event_dir) and data_source == 'Marsal2025':
                            event_sids = list(os.listdir(event_dir))
                        elif os.path.isdir(event_dir) and data_source == 'Marsal2021':
                            event_sids = [subdir[:3] for subdir in os.listdir(event_dir)]
                        else:
                            logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                            continue
                        info_file = 'marsal_2025_info.csv' if data_source == 'Marsal2025' else 'marsal_2021_info.csv'
                        file = os.path.join(data_dir, data_source.lower(), info_file)
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            next(rows)
                            for row in rows:
                                site_id = row[0]
                                if site_id.startswith(tuple(event_sids)):
                                    lat_val, _ = row[1].strip().split()
                                    geo_lat = float(lat_val[:2]) + float(lat_val[2:]) / 60.0
                                    lon_val, lon_dir = row[2].strip().split()
                                    geo_lon = float(lon_val[:3]) + float(lon_val[3:]) / 60.0
                                    if lon_dir in ['W', 'w']:
                                        geo_lon *= -1
                                    info_list = _add_info_row(info_list, site_id, geo_lat, geo_lon, data_type, data_class, data_source, event)
                    else:
                        logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")

            elif data_source == 'Zhang2020':
                for data_type in data_types:
                    if 'measured' in data_classes and data_type == 'GIC':
                        data_class = 'measured'
                        file_dir = os.path.join(data_dir, data_source.lower())
                        event_dir = os.path.join(file_dir, event, data_type.lower())
                        if not os.path.exists(event_dir):
                            logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                            continue
                        file = os.path.join(file_dir, 'zhang_2020_info.csv')
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            next(rows)
                            for row in rows:
                                info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)

            elif data_source == 'AlvesRibeiro':
                for data_type in data_types:
                    if data_type != 'GIC':
                        continue
                    file_dir = os.path.join(data_dir, data_source.lower())
                    event_dir = os.path.join(file_dir, event, data_type.lower())
                    if not os.path.exists(event_dir):
                        logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                        continue
                    file = os.path.join(file_dir, 'alvesribeiro_2023_info.csv')
                    with open(file, 'r') as csvfile:
                        rows = csv.reader(csvfile, delimiter=',')
                        next(rows)
                        for row in rows:
                            for data_class in ['measured', 'calculated']:
                                if data_class in data_classes:
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)

            elif data_source == 'Blake':
                for data_type in data_types:
                    if data_type != 'GIC':
                        continue
                    file_dir = os.path.join(data_dir, data_source.lower())
                    event_dir = os.path.join(file_dir, event, data_type.lower())
                    if not os.path.exists(event_dir):
                        logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                        continue
                    file = os.path.join(file_dir, 'blake_2018_info.csv')
                    with open(file, 'r') as csvfile:
                        rows = csv.reader(csvfile, delimiter=',')
                        next(rows)
                        for row in rows:
                            for data_class in ['measured', 'calculated']:
                                if data_class in data_classes:
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)

            elif data_source == 'Espinosa':
                for data_type in data_types:
                    if 'measured' in data_classes and data_type == 'GIC':
                        data_class = 'measured'
                        file_dir = os.path.join(data_dir, data_source.lower())
                        event_dir = os.path.join(file_dir, event, data_type.lower())
                        if not os.path.exists(event_dir):
                            logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                            continue
                        file = os.path.join(file_dir, 'espinosa_2019_info.csv')
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            next(rows)
                            for row in rows:
                                info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)
                    else:
                        logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")

            elif data_source == 'Bailey':
                for data_type in data_types:
                    if data_type == 'GIC':
                        file_dir = os.path.join(data_dir, data_source.lower())
                        event_dir = os.path.join(file_dir, event, data_type.lower())
                        if not os.path.exists(event_dir):
                            logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                            continue
                        file = os.path.join(file_dir, 'bailey_2022_info.csv')
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            next(rows)
                            for row in rows:
                                if 'measured' in data_classes:
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, 'measured', data_source, event)
                                if 'calculated' in data_classes:
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, 'calculated', data_source, event)
                    else:
                        logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")

            elif data_source == 'Nahayo':
                for data_type in data_types:
                    if 'calculated' in data_classes:
                        logger.info(f"   No calculated {data_type} data for Nahayo. Skipping...")
                    if data_type == 'GIC' and 'measured' in data_classes:
                        file_dir = os.path.join(data_dir, data_source.lower())
                        event_dir = os.path.join(file_dir, event, data_type.lower())
                        if not os.path.exists(event_dir):
                            logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                            continue
                        file = os.path.join(file_dir, event, 'nahayo_2022_info.csv')
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            next(rows)
                            for row in rows:
                                info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, 'measured', data_source, event)
                    else:
                        logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")

            elif data_source == 'Watari':
                for data_type in data_types:
                    if 'calculated' in data_classes:
                        logger.info(f"   No calculated {data_type} data for Watari. Skipping...")
                    if 'measured' in data_classes:
                        data_class = 'measured'
                        file_dir = os.path.join(data_dir, data_source.lower())
                        if data_type == 'GIC':
                            event_dir = os.path.join(file_dir, event, data_type.lower())
                        if data_type == 'B':
                            event_dir = os.path.join(file_dir, event, 'mag')
                        if not os.path.exists(event_dir):
                            logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                            continue
                        file = os.path.join(file_dir, 'watari_2009_info.csv')
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            next(rows)
                            for row in rows:
                                if row[3] == data_type:
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)
                    else:
                        logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")

            elif data_source == 'Caraballo2025':
                for data_type in data_types:
                    if data_type == 'B':
                        logger.info(f"   No B data is available for {data_source}. Skipping...")
                    if data_type == 'GIC':
                        event_dir = os.path.join(data_dir, data_source.lower(), event, 'gic')
                        if not os.path.exists(event_dir):
                            logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                            continue
                        file = os.path.join(data_dir, data_source.lower(), 'caraballo_2025_info.csv')
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            next(rows)
                            for row in rows:
                                if 'measured' in data_classes:
                                    data_class = 'measured'
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)
                                if 'calculated' in data_classes:
                                    data_class = 'calculated'
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)


            elif data_source == 'Caraballo2023':
                for data_type in data_types:
                    if data_type == 'GIC':
                        file_dir = os.path.join(data_dir, data_source.lower())
                        data_type_dir = os.path.join(file_dir, event[:4], data_type.lower())
                        if not os.path.exists(data_type_dir):
                            logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                            continue
                        data_file = os.path.join(data_type_dir, 'gic_2021ultproc.csv')
                        if not _event_overlaps_data(event, data_file, events, logger):
                            continue
                        file = os.path.join(file_dir, 'caraballo_2023_info.csv')
                        with open(file, 'r') as csvfile:
                            rows = csv.reader(csvfile, delimiter=',')
                            next(rows)
                            for row in rows:
                                if row[3] != data_type:
                                    continue
                                if 'measured' in data_classes:
                                    data_class = 'measured'
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)
                                if 'calculated' in data_classes:
                                    data_class = 'calculated'
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)
                    if data_type == 'B':
                        if 'calculated' in data_classes:
                            logger.info(f"   No calculated {data_type} data for {data_source}. Skipping...")
                        if 'measured' in data_classes:
                            data_class = 'measured'
                            file_dir = os.path.join(data_dir, data_source.lower())
                            data_type_dir = os.path.join(file_dir, event[:4], 'mag')
                            if not os.path.exists(data_type_dir):
                                logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                                continue
                            data_file = os.path.join(data_type_dir, 'tuc_magnetic_field.csv')
                            if not _event_overlaps_data(event, data_file, events, logger):
                                continue
                            file = os.path.join(file_dir, 'caraballo_2023_info.csv')
                            with open(file, 'r') as csvfile:
                                rows = csv.reader(csvfile, delimiter=',')
                                next(rows)
                                for row in rows:
                                    if row[3] != data_type:
                                        continue
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)

            elif data_source == 'Hubert':
                if 'DMM' not in data_types or 'measured' not in data_classes:
                    logger.info(f"   Only measured DMM data are available for data source {data_source}; skipping.")
                    continue
                file_dir = os.path.join(data_dir, data_source.lower())
                event_sites = _hubert_sites_for_event(file_dir, event, events)
                file = os.path.join(file_dir, 'hubert_2024_info.csv')
                with open(file, 'r') as csvfile:
                    rows = csv.reader(csvfile, delimiter=',')
                    next(rows)
                    for row in rows:
                        if row[0] in event_sites:
                            info_list = _add_info_row(
                                info_list, row[0], float(row[1]), float(row[2]),
                                'DMM', 'measured', data_source, event)

            elif data_source == 'Hughes':
                import json
                if 'GIC' not in data_types or 'measured' not in data_classes:
                    logger.info(f"   Only measured GIC data are available for data source {data_source}; skipping.")
                    continue
                file_dir = os.path.join(data_dir, data_source.lower())
                name_json = json.load(open(os.path.join(file_dir, 'hughes_site_names.json')))
                event_sites = set()
                for node, data_file in sorted(_hughes_files(file_dir).items()):
                    if _event_overlaps_data(event, data_file, events, logger):
                        site_id = [key for key, value in name_json.items() if value.get("node") == float(node[-1])]
                        if site_id:
                            event_sites.add(site_id[0])
                info_file = os.path.join(file_dir, 'hughes_2018_info.csv')
                with open(info_file, 'r') as csvfile:
                    rows = csv.reader(csvfile, delimiter=',')
                    next(rows)
                    for row in rows:
                        if row[0] in event_sites:
                            info_list = _add_info_row(
                                info_list, row[0], float(row[1]), float(row[2]),
                                'GIC', 'measured', data_source, event)
            else:
                raise ValueError(f"     Data source {data_source} not recognized.")

    info_df = pd.DataFrame(info_list)
    print(info_df)
    info_fname = CONFIG['files']['info']
    logger.info(f'   Saving info to {info_fname}')
    info_df.to_csv(info_fname, index=False)

def _source_available(data_source, data_dir, logger):
    if data_source == 'INTERMAG':
        return True

    source_dir = os.path.join(data_dir, data_source.lower())
    if not os.path.isdir(source_dir):
        logger.warning(f"   Data directory {source_dir} not found for data source {data_source}. Skipping...")
        return False
    return True


def _event_available(event, data_source, data_dir, logger):
    event_folder = os.path.join(data_dir, data_source.lower(), event)
    if not os.path.isdir(event_folder):
        logger.warning(f"   Event {event} not found for data source {data_source}. Skipping...")
        return False
    logger.info(f"   Adding sites for event {event} from data source {data_source} to info.csv.")
    return True


def _event_overlaps_data(event, data_file, events, logger):
    # Used for large, annual data files to parse if passed event is within available data
    import pandas as pd

    if not os.path.exists(data_file):
        raise ValueError(f"   Data file {data_file} not found.")


    if data_file not in _DATA_RANGE_CACHE:
        # Read first data row
        with open(data_file, 'r', newline='') as csvfile:
            rows = csv.reader(csvfile)
            next(rows)
            first_row = next(rows, None)
        # Read last data row
        with open(data_file, 'rb') as csvfile:
            csvfile.seek(0, os.SEEK_END)
            file_size = csvfile.tell()
            read_size = min(file_size, 8192)
            csvfile.seek(-read_size, os.SEEK_END)
            lines = csvfile.read().decode('utf-8').splitlines()
            last_line = next((line for line in reversed(lines) if line.strip()), None)

        if first_row is None or last_line is None:
            logger.warning(f"   No data rows found in {data_file}. Skipping...")
            return False
        last_row = next(csv.reader([last_line]))
        dates = pd.to_datetime([first_row[0], last_row[0]], errors='coerce').dropna()
        if len(dates) != 2:
            logger.warning(f"   Could not determine the date range in {data_file}. Skipping...")
            return False
        _DATA_RANGE_CACHE[data_file] = (dates.min(), dates.max())

    # Using start/stop times for config to determine if event overlaps w annual dataset
    event_start, event_stop = events[event]['data_limits']
    available_start, available_stop = _DATA_RANGE_CACHE[data_file]
    if pd.Timestamp(event_stop) < available_start or pd.Timestamp(event_start) > available_stop:
        logger.warning(
            f"   Event {event} does not overlap {available_start} through "
            f"{available_stop} for {os.path.basename(data_file)}. Skipping..."
        )
        return False
    return True


def _hubert_sites_for_event(source_dir, event, events):
    import json
    # Define start/stop from event
    event_start, event_stop = events[event]['data_limits']
    event_start = event_start.date() if hasattr(event_start, 'date') else datetime.date.fromisoformat(event_start[:10])
    event_stop = event_stop.date() if hasattr(event_stop, 'date') else datetime.date.fromisoformat(event_stop[:10])
    # Read Hübert site names from JSON file
    event_sites = set()
    site_names_file = os.path.join(source_dir, 'hubert_site_names.json')
    with open(site_names_file, 'r') as jsonfile:
        site_names = json.load(jsonfile)
    # Check for available event data for each site
    for site_id, site_info in site_names.items():
        site_number = site_info.get('number')
        if site_number is None:
            continue
        file_pattern = re.compile(rf'^{site_id}([RU])_(\d{{4}}_\d{{2}}_\d{{2}})\.xyz$')
        # Looking for Hübert data within event date range
        for year in range(event_start.year, event_stop.year + 1):
            site_dir = os.path.join(source_dir, f'{year}', 'dmm', f'{site_number}')
            if not os.path.isdir(site_dir):
                continue
            for filename in os.listdir(site_dir):
                match = file_pattern.match(filename)
                if match is None:
                    continue
                file_date = datetime.datetime.strptime(match.group(2), '%Y_%m_%d').date()
                if event_start <= file_date <= event_stop:
                    site_type = 'ref' if match.group(1) == 'R' else 'line'
                    event_sites.add(f"{site_id} {site_type}")
    # Return the set of sites with available data for the event
    return event_sites


def _hughes_files(source_dir):
    files = {}
    file_pattern = re.compile(r'^Node_(\d+)_.*\.csv$')
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            match = file_pattern.match(filename)
            if match is not None:
                files[f"Node {match.group(1)}"] = os.path.join(root, filename)
    return files