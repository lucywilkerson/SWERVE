# Update info.extended.csv and info.extended.json with automated errors
# Only used in main.py
import json

def update_info_extended(sids_only, data, logger=None, CONFIG=None):
  from swerve import read_info_df, infodf2dict
  if CONFIG is None:
    from swerve import config
    CONFIG = config()
  info_df = read_info_df(extended=True, logger=logger)
  for sid in sids_only:
    if 'GIC' in data[sid] and 'measured' in data[sid]['GIC'].keys():
      for data_source in data[sid]['GIC']['measured'].keys():
        error_msg = data[sid]['GIC']['measured'][data_source][sid]['automated_error']
        logger.info(f"  Adding error for site '{sid}', GIC/'measured/{data_source}: {error_msg}")
        info_df.loc[(info_df['site_id'] == sid) & (info_df['data_type'] == 'GIC') & (info_df['data_class'] == 'measured'), 'automated_error'] = error_msg
  out_fname = CONFIG['files']['info_extended']
  info_df.to_csv(out_fname, index=False)
  logger.info(f"Wrote {out_fname}")

  print(f"Preparing {CONFIG['files']['info_extended_json']}")
  info_dict = infodf2dict(info_df, logger)

  logger.info(f"Writing {CONFIG['files']['info_extended_json']}")
  with open(CONFIG['files']['info_extended_json'], 'w') as f:
    json.dump(info_dict, f, indent=2)