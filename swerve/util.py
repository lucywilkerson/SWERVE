import argparse
import csv
import json
import os
import pickle
import shutil
import zipfile


def unzip():
    """Unzip all configured NERC GIC and magnetometer archives."""
    from swerve import config

    def _unzip_all_zip_files(directory_path, output_directory=None):
        """Unzip and remove all zip files found in a directory."""
        for item in os.listdir(directory_path):
            if not item.endswith(".zip"):
                continue
            zip_file_path = os.path.join(directory_path, item)
            if output_directory:
                extraction_path = output_directory
            else:
                zip_name_without_extension = os.path.splitext(item)[0]
                extraction_path = os.path.join(directory_path, zip_name_without_extension)
                os.makedirs(extraction_path, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
                print(f"Successfully unzipped '{item}' to '{extraction_path}'")
                os.remove(zip_file_path)
                print(f"Deleted original zip file: '{item}'.")
            except zipfile.BadZipFile:
                print(f"Error: '{item}' is a bad zip file and cannot be unzipped.")
            except Exception as error:
                print(f"An error occurred while unzipping '{item}': {error}")

    CONFIG = config()
    data_dir = CONFIG['dirs']['data']
    events = CONFIG['event']
    for nerc_data_type in ['gic', 'mag']:
        for event in events:
            print(f'Unzipping NERC files for event {event}')
            zip_dir = os.path.join(data_dir, 'data_original', 'nerc', event, nerc_data_type)
            _unzip_all_zip_files(zip_dir, zip_dir)


def move_pics():
    """Collect configured NERC measured-GIC plots into each event summary directory."""
    from swerve import config, sids, sids_and_events

    CONFIG = config()
    data_dir = CONFIG['dirs']['data']
    sids_only = None
    if len(CONFIG['event']) > 1:
        sids_only = sids_and_events(key=sids_only, data_type='GIC', data_class='measured')
    else:
        sids_only = sids(key=sids_only, data_type='GIC', data_class='measured', add_event=True)

    for sid, event in sids_only:
        event_dir = os.path.join(data_dir, 'data_processed', event)
        sid = sid.lower().replace(' ', '')
        source_image = os.path.join(event_dir, 'sites', sid, 'figures', 'original', 'GIC_measured_NERC.png')
        if not os.path.isfile(source_image):
            continue
        destination_folder = os.path.join(event_dir, '_all', 'all_gic')
        new_fname = f'{sid}_GIC_measured.png'
        os.makedirs(destination_folder, exist_ok=True)

        try:
            shutil.copy(source_image, os.path.join(destination_folder, new_fname))
            print(f"'{source_image}' copied successfully to '{destination_folder}'")
        except FileNotFoundError:
            print(f"Error: Source file '{source_image}' not found.")
        except Exception as error:
            print(f"An error occurred: {error}")


def _write_pkl(fname, data, logger, indent=''):
    from swerve import config

    CONFIG = config()
    fname = os.path.join(CONFIG['dirs']['data'], fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as file_handle:
        logger.info(f"{indent}Writing {fname}")
        pickle.dump(data, file_handle)


def update_info_extended(sids_only, data, exclude_errors=None, logger=None, CONFIG=None):
    from swerve import infodf2dict, read_info_df

    if CONFIG is None:
        from swerve import config
        CONFIG = config()
    info_df = read_info_df(extended=True, exclude_errors=exclude_errors, logger=logger)
    for sid, event in sids_only:
        if 'GIC' in data[sid] and 'measured' in data[sid]['GIC'].keys():
            for data_source in data[sid]['GIC']['measured'].keys():
                error_msg = data[sid]['GIC']['measured'][data_source][sid]['automated_error']
                if error_msg is not None:
                    logger.info(f"  Adding error for site '{sid}', GIC/'measured/{data_source}: {error_msg}")
                    mask = ((info_df['site_id'] == sid)
                            & (info_df['event'] == event)
                            & (info_df['data_type'] == 'GIC')
                            & (info_df['data_class'] == 'measured'))
                    info_df.loc[mask, 'automated_error'] = str(error_msg)
    out_fname = CONFIG['files']['info_extended']
    info_df.to_csv(out_fname, index=False)
    logger.info(f"Wrote {out_fname}")

    logger.info(f"Preparing {CONFIG['files']['info_extended_json']}")
    info_dict = infodf2dict(info_df, logger)
    logger.info(f"Writing {CONFIG['files']['info_extended_json']}")
    with open(CONFIG['files']['info_extended_json'], 'w') as file_handle:
        json.dump(info_dict, file_handle, indent=2)


def write_info_csv():
    """Write the configured site metadata to info.csv."""
    import pandas as pd
    from swerve import config

    def _check_event(event, data_source, data_dir, logger):
        event_folder = os.path.join(data_dir, data_source.lower(), event)
        if not os.path.exists(event_folder):
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
            if data_source == 'NERC':
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
                            file_dir = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), 'GIC-measured')
                            file = os.path.join(file_dir, 'GIC_monitors.dat')
                            if not os.path.exists(file):
                                raise FileNotFoundError(f"File not found: {file}")
                            with open(file, 'r') as csvfile:
                                rows = csv.reader(csvfile, delimiter=',')
                                next(rows)
                                for row in rows:
                                    info_list = _add_info_row(info_list, row[0], float(row[2]), float(row[3]), data_type, data_class, data_source, event)
                        elif data_type == 'B':
                            file_dir = os.path.join(data_dir, data_source.lower(), event, 'mag')
                            file = os.path.join(file_dir, 'TVAmagmetadata.dat')
                            if not os.path.exists(file):
                                raise FileNotFoundError(f"File not found: {file}")
                            with open(file, 'r') as csvfile:
                                rows = csv.reader(csvfile, delimiter=',')
                                for row in rows:
                                    info_list = _add_info_row(info_list, row[0], float(row[1]), float(row[2]), data_type, data_class, data_source, event)
                        else:
                            raise ValueError(f"Data type not recognized: {data_type}. Valid options are 'GIC' and 'B'.")
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

            elif data_source in ['AlvesRibeiro', 'Blake']:
                info_name = 'alvesribeiro_2023_info.csv' if data_source == 'AlvesRibeiro' else 'blake_2018_info.csv'
                for data_type in data_types:
                    if data_type != 'GIC':
                        continue
                    file_dir = os.path.join(data_dir, data_source.lower())
                    event_dir = os.path.join(file_dir, event, data_type.lower())
                    if not os.path.exists(event_dir):
                        logger.warning(f"   Data type {data_type} not found for source {data_source} and event {event}. Skipping...")
                        continue
                    file = os.path.join(file_dir, info_name)
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
            else:
                raise ValueError(f"     Data source {data_source} not recognized.")

    info_df = pd.DataFrame(info_list)
    print(info_df)
    info_fname = CONFIG['files']['info']
    logger.info(f'   Saving info to {info_fname}')
    info_df.to_csv(info_fname, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run SWERVE file utilities.')
    parser.add_argument('command', choices=['unzip', 'move-pics'])
    args = parser.parse_args(argv)
    if args.command == 'unzip':
        unzip()
    else:
        move_pics()


if __name__ == '__main__':
    main()