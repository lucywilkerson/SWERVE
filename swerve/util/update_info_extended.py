import json


def update_info_extended(sids_only, data, exclude_errors=None, logger=None, CONFIG=None):
    from swerve import infodf2dict, read_info_df

    if CONFIG is None:
        from swerve import config
        CONFIG = config()
    info_df = read_info_df(extended=True, exclude_errors=exclude_errors, logger=logger)
    for sid, event in sids_only:
        site_data = data[event][sid]
        if 'GIC' in site_data and 'measured' in site_data['GIC'].keys():
            for data_source in site_data['GIC']['measured'].keys():
                error_msg = site_data['GIC']['measured'][data_source][sid]['automated_error']
                if error_msg is not None:
                    logger.info(f"  Adding error for site '{sid}', GIC/'measured/{data_source}: {error_msg}")
                    mask = ((info_df['site_id'] == sid)
                            & (info_df['event'] == event)
                            & (info_df['data_type'] == 'GIC')
                            & (info_df['data_class'] == 'measured')
                            & (info_df['data_source'] == data_source))
                    info_df.loc[mask, 'automated_error'] = str(error_msg)
    out_fname = CONFIG['files']['info_extended']
    info_df.to_csv(out_fname, index=False)
    logger.info(f"Wrote {out_fname}")

    logger.info(f"Preparing {CONFIG['files']['info_extended_json']}")
    info_dict = infodf2dict(info_df, logger)
    logger.info(f"Writing {CONFIG['files']['info_extended_json']}")
    with open(CONFIG['files']['info_extended_json'], 'w') as file_handle:
        json.dump(info_dict, file_handle, indent=2)