import os
import csv
import numpy
import pandas
import pickle
import datetime

debug = False  # Set to True to log resampling information.

def site_read(sid, data_types=None, reparse=False, add_errors=False, logger=None, debug=False):
  """Read data from one or more sites

  Usage:
    site_read(sid, data_types=None, reparse=False, logger=None):

  If `data_types` is None, read all data types (e.g, B, GIC) for the site.

  If 'add_errors' is True, add automated error checks to data.

  If `reparse` is True, reparse the data files even if cache file exists
  (use if data files or code in this script that reads them has changed).

  If `debug` is True, print processing details for computing resampled data.
  """
  from swerve import config, read_info_dict, resample

  CONFIG = config()

  if logger is None:
    logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

  resample_kwargs = {}
  if debug:
    resample_kwargs = {'logger': logger, 'logger_indent': 6}

  sidx = sid.lower().replace(' ', '')
  site_all_file = '_all.pkl'
  out_dir = CONFIG['dirs']['processed']
  site_all_file = os.path.join(CONFIG['dirs']['data'], out_dir, sidx, 'data', site_all_file)

  logger.info(f"Reading '{sid}' data")

  if not reparse:
    if os.path.exists(site_all_file):
      logger.info(f"  Reading cached file with all data for site '{sid}': {site_all_file}")
      with open(site_all_file, 'rb') as f:
        data = pickle.load(f)
        return data

  start = CONFIG['limits']['data'][0]
  stop = CONFIG['limits']['data'][1]

  site_info = read_info_dict(sid=sid)

  for data_type in site_info.keys(): # e.g., GIC, B

    if data_types is not None and data_type not in data_types:
      # Skip this data type if not in requested data_types to plot.
      logger.info(f"  Not reading '{sid}/{data_type}' data type data b/c not in requested data_types = {data_types}.")
      continue

    if data_type not in site_info:
      # This will occur if data_types is given and site does not have a
      # data_type in data_types.
      logger.warning(f"  Requesested data_type = {data_type} not available at site '{sid}'. Skipping.")
      continue

    data_classes = site_info[data_type].keys()

    for data_class in data_classes: # e.g., measured, calculated

      data_sources = site_info[data_type][data_class]

      for data_source in data_sources.keys():

        logger.info(f"  Reading '{data_type}/{data_class}/{data_source}' data")
        orig = _site_read_orig(sid, data_type, data_class, data_source, logger)
        if sid in CONFIG['single_phase_sids'] and data_type == 'GIC':
          logger.info(f"    Multiplying GIC data by 3 to account for single-phase transformer.")
          orig['data'] = orig['data'] * 3
        site_info[data_type][data_class][data_source]['original'] = orig
        # Check returned data object
        if _output_error(orig, logger):
          site_info[data_type][data_class][data_source][sid]['automated_error'] = orig['error']
          continue

        if 'automated_error' not in site_info[data_type][data_class][data_source][sid].keys():
          add_errors = True
        if add_errors and data_type == 'GIC' and data_class == 'measured':
          from swerve import find_errors
          logger.info('    Running automated error checks on GIC measured data')
          site_info[data_type][data_class][data_source][sid]['automated_error'] = find_errors(orig, sid, data_source)

        resample_msg = "Resample to 1m aves and NaN pad or trim to start/stop."
        data_mod = orig['data'].copy()
        if data_type == 'B' and data_class == 'measured':
            logger.info(f'    Remove baseline then {resample_msg}')
            for i in range(3):
              # TODO: Get IGRF value instead of using first value?
              first_valid_idx = numpy.where(~numpy.isnan(orig["data"][:, i]))[0][0]
              baseline = orig["data"][first_valid_idx, i]
              data_mod[:,i] = orig["data"][:, i] - baseline
        else:
          logger.info(f"    {resample_msg}")

        labels = orig['labels']
        if data_type == 'B':
          # Add column with horizontal magnitude
          h = data_mod[:,0]**2 + data_mod[:,1]**2
          data_mod = numpy.hstack((data_mod, numpy.sqrt(h).reshape(-1, 1)))
          labels.append('B_H')

        modified = {'modification': resample_msg}
        try:
          time_m, data_m = resample(orig["time"], data_mod, start, stop, ave='60s', **resample_kwargs)
          modified['time'] = time_m
          modified['data'] = data_m
          modified['unit'] = orig['unit']
          modified['labels'] = labels
        except Exception as e:
          modified['error'] = str(e)
          logger.error(f"    Error resampling data: {modified['error']}")

        site_info[data_type][data_class][data_source]['modified'] = modified

        file_name = f'{data_type}_{data_class}_{data_source}.pkl'
        file_name = os.path.join(CONFIG['dirs']['processed'], sidx, 'data', file_name)
        _write_pkl(file_name, site_info[data_type][data_class], logger, indent= ' '*4)

  if data_types is None:
    _write_pkl(site_all_file, site_info, logger, indent=' '*2)
  else:
    logger.warning(f"  Not writing {site_all_file} b/c data_types is not None.")
    if reparse == True and data_types is not None and not sid.lower().startswith('test'):
      raise ValueError("Reparse requested with data_types not None, which is not supported unless running for test data.")

  return site_info

def _site_read_orig(sid, data_type, data_class, data_source, logger):

  """Read data from site

  Returns:
      dict with keys
        time:   1D numpy array of datetimes
        data:   2D numpy array of data; for B or dB data, the shape is (N, 3);
                for GIC measurements, the shape is (N, 1), where N = len(time).
        label:  For B (or dB) a 3-element list of labels used by data
                provider.
        unit:   String with unit used by data provider that applies to all
                columns in data.

      and for MAGE B data only:
        data_raw:  Original columns of data as read from file.
        label_raw: Original column labels as read from file.
  """

  def read_nerc(data_dir, fname):
    data = []
    time = []

    file = os.path.join(data_dir, fname)
    logger.info(f"    Reading {file}")
    if not os.path.exists(file):
      raise FileNotFoundError(f"File not found: {file}")
    with open(file, 'r') as csvfile:
      rows = csv.reader(csvfile, delimiter=',')
      next(rows)  # Skip header row.
      device_id_last = None
      for row in rows:
        device_id = row[0]
        if device_id != device_id_last:
          if device_id_last is not None:
            raise ValueError(f"Multiple device ids found in {file}")
          device_id_last = device_id
        time_p = datetime.datetime.strptime(row[1], '%m/%d/%Y %I:%M:%S %p')
        time.append(time_p)
        cols = []
        for i in range(2, len(row)):
          cols.append(float(row[i]))
        data.append(cols)

    time = numpy.array(time)
    data = numpy.array(data)

    ret = {"time": time, "data": data}
    if len(time) != len(numpy.unique(time)):
      # e.g., 2024E04_10233.csv, which looks like it is 10-second cadence data
      # but time stamps have the same second value.
      ret['error'] = 'Duplicate time stamps found'
      #logger.info(time[numpy.where(numpy.diff(time) == datetime.timedelta(0))])

    return ret

  from swerve import config
  CONFIG = config()
  data_dir = CONFIG['dirs']['data']

  data = []
  time = []

  if data_type == 'GIC' and data_class == 'measured' and data_source == 'TVA':
    data_dir = os.path.join(data_dir, 'tva', 'gic', 'GIC-measured')
    sid = sid.lower().replace(' ','')
    if sid == 'widowscreek':
      sid = f'{sid}2'
    fname = f'gic-{sid}_20240510.csv'
    file = os.path.join(data_dir, fname)
    logger.info(f"    Reading {file}")
    if not os.path.exists(file):
      raise FileNotFoundError(f"File not found: {file}")
    with open(file, 'r') as csvfile:
      rows = csv.reader(csvfile, delimiter=',')
      for row in rows:
          time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
          data.append(float(row[1]))

    # Reshape to 2D array with a single column
    data = numpy.array(data).reshape(-1, 1)
    return {
      "time": numpy.array(time).flatten(),
      "data": data,
      "labels": ["GIC"],
      "unit": "A",
    }

  if data_type == 'GIC' and data_class == 'measured' and data_source == 'NERC':
    nerc_prefix = CONFIG['nerc_prefix']
    fname = f'{nerc_prefix}_{sid}.csv'
    data_dir = os.path.join(data_dir, 'nerc', 'gic')
    data = read_nerc(data_dir, fname)
    return {**data, "labels": ["GIC"], "unit": "A"}

  if data_type == 'GIC' and data_class == 'calculated' and data_source == 'TVA':
    data_dir = os.path.join(data_dir, 'tva', 'gic', 'GIC-calculated')
    sid = sid.replace(' ','')
    if sid == 'BullRun':
      sid = 'BullRunXfrm' # BullRun file Xfrm appended to name in file name.
    if sid == 'WidowsCreek':
      sid = f'{sid}2'

    dates = ['20240510', '20240511', '20240512']
    time = []
    data = []
    for date in dates:
      file = f'{date}_{sid}GIC.dat'
      file = os.path.join(data_dir, file)
      logger.info(f"    Reading {file}")
      if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
      d, times = numpy.loadtxt(file, unpack=True, skiprows=1, delimiter=',')
      data.append(d)
      dto = datetime.datetime.strptime(date, '%Y%m%d')
      for t in times:
        time.append(dto + datetime.timedelta(seconds=t))

    # Reshape to 2D array with a single column.
    data = numpy.array(data).reshape(-1, 1)
    return {
      "time": numpy.array(time).flatten(),
      "data": data,
      "labels": ["GIC"],
      "unit": "A"
    }

  if data_type == 'GIC' and data_class == 'calculated' and data_source == 'GMU':
    from swerve import read_info_df, read_info_dict

    extended_df = read_info_df(extended=True)
    query = (extended_df['site_id'] == sid) & (extended_df['data_source'] == 'GMU')
    nearest_sim_site = extended_df.loc[query, 'nearest_sim_site']
    nearest_sim_site = int(nearest_sim_site.values[0])
    logger.info(f"      Nearest simulation site: {nearest_sim_site}")

    info = read_info_dict(sid)
    measured_sources = [source for source in info['GIC']['measured'] if isinstance(source, str)]
    if 'NERC' in measured_sources:
      fname = os.path.join(data_dir, 'gmu', 'nerc', f'site_{nearest_sim_site}.csv')
    elif 'TVA' in measured_sources:
      fname = os.path.join(data_dir, 'gmu', 'tva', f'site_{nearest_sim_site}.csv')
    else:
      raise ValueError(f"No corresponding measured data source found for site {sid}")
    logger.info(f"      Reading {fname}")
    if not os.path.exists(fname):
      raise FileNotFoundError(f"File not found: {fname}")

    data = []
    time = []

    with open(fname,'r') as csvfile:
      rows = csv.reader(csvfile, delimiter = ',')
      next(rows)  # Skip header row.
      for row in rows:
        time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
        #data.append([float(row[2]), float(row[3]), float(row[4])])
        data.append([float(row[2])])

    data = numpy.array(data)
    time = numpy.array(time)
    return {"time": time, "data": data, "labels": ["GIC"], "unit": "A"}

  if data_type == 'B' and data_class == 'measured' and data_source == 'TVA':
    data_dir = os.path.join(data_dir, 'tva', 'mag')
    sid = sid.lower().replace(' ','')

    data  = []
    time = []

    file = os.path.join(data_dir, f'{sid}_mag_20240509.csv')
    logger.info(f"    Reading {file}")
    if not os.path.exists(file):
      raise FileNotFoundError(f"File not found: {file}")

    with open(file,'r') as csvfile:
      rows = csv.reader(csvfile, delimiter = ',')
      next(rows)  # Skip header row.
      for row in rows:
        time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
        data.append([float(row[1]), float(row[2]), float(row[3])])

    data = numpy.array(data)
    time = numpy.array(time)
    return {"time": time, "data": data, "labels": ["Bx", "By", "Bz"], "unit": "nT"}

  if data_type == 'B' and data_class == 'measured' and data_source == 'NERC':
    # TODO: magnetometers.csv indicates if GEO or MAG coordinates
    nerc_prefix = CONFIG['nerc_prefix']
    fname = f'{nerc_prefix}_{sid}.csv'
    data_dir = os.path.join(data_dir, 'nerc', 'mag')
    data = read_nerc(data_dir, fname)
    return {**data, "labels": ["B_N", "B_E", "B_v"], "unit": "nT"}

  if data_type == 'B' and data_class == 'calculated' and data_source in ['SWMF', 'OpenGGCM']:

    sid = sid.replace(' ','')
    data_dir = os.path.join(data_dir, data_source.lower(), sid.lower())

    if data_source == 'OpenGGCM':
      file = os.path.join(data_dir, f'dB_{data_source}_{sid}.pkl')
    else:
      file = os.path.join(data_dir, f'dB_{sid}.pkl')

    logger.info(f"    Reading {file}")
    if not os.path.exists(file):
      raise FileNotFoundError(f"File not found: {file}")
    df  = pandas.read_pickle(file)

    bx = df['Bn_msph'] + df['Bn_gap'] + df['Bnh_iono'] + df['Bnp_iono']
    by = df['Be_msph'] + df['Be_gap'] + df['Beh_iono'] + df['Bep_iono']
    bz = df['Bd_msph'] + df['Bd_gap'] + df['Bdh_iono'] + df['Bdp_iono']

    data = numpy.vstack([bx.to_numpy(), by.to_numpy(), bz.to_numpy()])
    time = bx.keys() # Will be the same for all
    time = time.to_pydatetime()

    return {"time": time, "data": data.T, "labels": ["Bx", "By", "Bz"], "unit": "nT"}

  if data_type == 'B' and data_class == 'calculated' and data_source == 'MAGE':
    # TODO: A single file has data from all sites. Here we read full
    # file and only keep data for the requested site. Modify this code
    # so sites dict is cached and used if found.
    file = os.path.join(data_dir, 'mage', 'TVAinterpdf.csv')
    logger.info(f"    Reading {file}")
    if not os.path.exists(file):
      raise FileNotFoundError(f"File not found: {file}")

    sites = {}
    with open(file,'r') as csvfile:
      rows = csv.reader(csvfile, delimiter = ',')
      next(rows)  # Skip header row.
      for row in rows:
        site = row[0]
        if site not in sites:
          sites[site] = {"time": [], "data": [], "data_raw": []}
        sites[site]["time"].append(datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S'))
        # Header is
        # site,time,dBn,dBt,dBp,dBr,glon,glat,mlon,mlat
        # column #s:
        #   0,   1,  2,  3,  4,  5,   6,   7,   8,   9
        #
        # From Mike Wiltberger:
        # As a reminder I will point you to our documentation for the kaipy package which
        # includes information on the structure of SuperMAGE interpolated dataframe.
        # Here's the summary from that documentation.
        # - dBn: Interpolated northward deflection (dot product of dB and minus the theta unit vector)
        # - dBt: Interpolated magnetic theta component.
        # - dBp: Interpolated magnetic phi component.
        # - dBr: Interpolated magnetic radial component
        # It doesn't help that we didn't use a consistent naming convention with the SuperMAG data.
        # Here's the relevant mapping BNm - dBt, BEm - dBp, and BZm - dBr.
        # I've attached a reference plot as an example to this message.
        #
        # SuperMAG coordinate system description:
        #   https://supermag.jhuapl.edu/mag/?fidelity=low&tab=description
        #   Note that geomagnetic coordinates are routinely labeled HDZ although
        #   the units of the D-component can be nT or an angle. Likewise, the
        #   D-component is often found to have a significant offset. As a
        #   consequence SuperMAG decided to denote the components: B=(BN,BE,BZ)
        #     N-direction is local magnetic north
        #     E-direction is local magnetic east
        #     Z-direction is vertically down
        #
        # RSW comment:
        # I think "theta component" means theta_hat component. Similar for phi.
        # The MAGE coordinate systems seems be spherical with an origin at
        # Earth's center and the theta=0 along dipole axis.
        # The reference plot seems to equate BNm with dBn and not dBt.
        # In the following, we use the mapping implied by the plot.
        #                              BNm/dBn        BEm/dBp         BZm/dBr
        # 
        # RSW: Update based on communication with Kareem via Eric Winter:
        # dBn is the the northward deflection in _geomagnetic_ coordinates.
        # dBn is the component of the magnetic field in the direction of
        # geomagnetic north (and not a deflection in the sense of an angle, which
        # is commonly used in the magnetometer community).
        #
        # The r,phi,theta are in a spherical geographic coordinate system. (I
        # thought geomagnetic initially because of the word "magnetic"
        # in their definitions, which could be interpreted as meaning magnetic
        # coordinate system. Also, Mike's "mapping" statement above has dBr, dBp, and dBt
        # equated to SuperMAG's BZm, BEm, and BNm, which are in local _geomagnetic_.)

        for i in range(2, 6):
          row[i] = float(row[i])

        labels_raw = ["dBn", "dBt", "dBp", "dBr"]
        sites[site]["data_raw"].append(row[2:6])

        labels = ["-dBt", "dBp", "-dBr"]
        sites[site]["data"].append([-float(row[3]), float(row[4]), -float(row[5])])


    if sid not in sites:
      msg = f"Requested site name = '{sid}' associated with site id = '{sid}' not found in {file}"
      raise ValueError(msg)

    time = numpy.array(sites[sid]["time"])
    data = numpy.array(sites[sid]["data"])
    data_raw = numpy.array(sites[sid]["data_raw"])
    return {
      "time": time,
      "data": data,
      "data_raw": data_raw,
      "labels": labels,
      "labels_raw": labels_raw,
      "unit": "nT"
    }

  if data_type =='GIC' and data_source == 'TEST':
    fname = f'{sid}_{data_type}_{data_class}_timeseries.csv'
    data_dir = os.path.join(data_dir, 'test')

    data  = []
    time = []

    file = os.path.join(data_dir, fname)
    logger.info(f"    Reading {file}")
    if not os.path.exists(file):
      raise FileNotFoundError(f"File not found: {file}")

    with open(file, 'r') as csvfile:
      next(csvfile)  # Skip header row
      rows = csv.reader(csvfile, delimiter=',')
      for row in rows:
          time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
          data.append(float(row[1]) if row[1] != '' else numpy.nan)

    # Reshape to 2D array with a single column
    data = numpy.array(data).reshape(-1, 1)
    return {
      "time": numpy.array(time).flatten(),
      "data": data,
      "labels": ["GIC"],
      "unit": "A"
    }

  if data_type =='B' and data_source == 'TEST':
    fname = f'{sid}_{data_type}_{data_class}_timeseries.csv'
    data_dir = os.path.join(data_dir, 'test')

    data  = []
    time = []

    file = os.path.join(data_dir, fname)
    logger.info(f"    Reading {file}")
    if not os.path.exists(file):
      raise FileNotFoundError(f"File not found: {file}")

    with open(file, 'r') as csvfile:
      next(csvfile)  # Skip header row
      rows = csv.reader(csvfile, delimiter=',')
      for row in rows:
          time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
          data.append([float(row[1]), float(row[2]), float(row[3])])

    return {
      "time": numpy.array(time),
      "data": numpy.array(data),
      "labels": ["Bx", "By", "Bz"],
      "unit": "nT"
    }

def _output_error(d, logger):
  msgo = "Not computing modified"
  if 'error' in d:
    logger.error(f"    {msgo} due to error: {d['error']}")
    return True

  if len(d['data'].shape) != 2:
    logger.error(f"    {msgo} b/c data array is not 2D")
    return True

  if len(d['time'].shape) != 1:
    logger.error(f"    {msgo} b/c time array is not 1D")
    return True

  if d['data'].shape[0] != len(d['time']):
    msg = f"    {msgo} b/c d['data'].shape[0] = {d['data'].shape[0]} != len(d['time'])"
    msg += f" = {len(d['time'])}"
    logger.error(msg)
    return True

  return False

def _write_pkl(fname, data, logger, indent=''):

  import os
  from swerve import config
  CONFIG = config()
  fname = os.path.join(CONFIG['dirs']['data'], fname)
  if not os.path.exists(os.path.dirname(fname)):
    os.makedirs(os.path.dirname(fname))

  with open(fname, 'wb') as f:
    logger.info(f"{indent}Writing {fname}")
    pickle.dump(data, f)
