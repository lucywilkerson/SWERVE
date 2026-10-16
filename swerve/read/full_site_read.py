import os
import pickle

from .orig_readers import _site_read_orig
from ..util import write_pkl


def _full_output_file(processed_dir, data_source, partition, data_type, data_class, sid):
  filename = f"{data_type.lower()}_{data_class.lower()}_{sid.lower().replace(' ', '')}.pkl"
  path = os.path.join(processed_dir, 'full', data_source.lower())
  if partition != 'unpartitioned':
    path = os.path.join(path, partition)
  return os.path.join(path, data_type.lower(), filename)


def _site_read_full(sid, data_types, reparse, start, stop, logger, CONFIG):
  from swerve import read_info_df
  from swerve.util.source_catalog import source_kind, source_partitions

  info_kwargs = CONFIG['info_kwargs'].copy()
  if data_types is not None:
    info_kwargs['data_type'] = data_types
  info_df = read_info_df(extended=True, logger=logger, **info_kwargs)
  info_df = info_df[info_df['site_id'] == sid]
  if info_df.empty:
    raise ValueError(f"sid '{sid}' has no data matching the full-ingestion configuration.")

  data = {}
  identity_columns = ['site_id', 'data_type', 'data_class', 'data_source']
  original_dir = CONFIG['dirs']['original']
  processed_dir = CONFIG['dirs']['processed']

  for data_source in sorted(info_df['data_source'].dropna().unique()):
    source_info = info_df[info_df['data_source'] == data_source]
    kind = source_kind(data_source)
    partitions = source_partitions(original_dir, data_source)
    if data_source == 'INTERMAG':
      partitions = ['unpartitioned']

    for partition in partitions:
      if kind == 'event':
        partition_info = source_info[source_info['event'].astype(str) == partition]
      else:
        partition_info = source_info
      partition_info = partition_info.drop_duplicates(subset=identity_columns)

      for row in partition_info.to_dict('records'):
        data_type = str(row['data_type'])
        data_class = str(row['data_class'])
        reader_event = partition if kind != 'continuous' else 'full'
        output_file = _full_output_file(
          processed_dir,
          data_source,
          partition,
          data_type,
          data_class,
          sid,
        )

        logger.info(f"  Reading full '{data_type}/{data_class}/{data_source}' data for partition '{partition}'")
        if not reparse and os.path.exists(output_file):
          logger.info(f"    Reading cached file: {output_file}")
          with open(output_file, 'rb') as file_handle:
            original = pickle.load(file_handle)
        else:
          original = _site_read_orig(
            sid,
            data_type,
            data_class,
            data_source,
            reader_event,
            logger,
            start=start,
            stop=stop,
          )
          if original is None:
            raise NotImplementedError(
              f"No reader for {data_type}/{data_class}/{data_source}."
            )
          if sid in CONFIG['single_phase_sids'] and data_type == 'GIC':
            original['data'] = original['data'] * 3
          write_pkl(output_file, original, logger, indent=' '*4)

        data.setdefault(data_source, {}).setdefault(partition, {}).setdefault(data_type, {})[data_class] = original

  return data