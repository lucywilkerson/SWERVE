import csv
import os

def write_info_csv():
    """Write the configured site metadata to info.csv."""
    import pandas as pd
    from swerve import config, read_intermag_info

    def _check_event(event, data_source, data_dir, logger):
        event_folder = os.path.join(data_dir, data_source.lower(), event)
        if not os.path.exists(event_folder) and data_source != 'INTERMAG':
            logger.warning(f"   Event {event} not found for data source {data_source}. Skipping...")
            return False
        logger.info(f"   Adding sites for event {event} from data source {data_source} to info.csv.")
        return True

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
        for data_source in data_sources:
            if not _check_event(event, data_source, data_dir, logger):
                continue
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
            else:
                raise ValueError(f"     Data source {data_source} not recognized.")

    info_df = pd.DataFrame(info_list)
    print(info_df)
    info_fname = CONFIG['files']['info']
    logger.info(f'   Saving info to {info_fname}')
    info_df.to_csv(info_fname, index=False)

