import os
import csv
import numpy
import pandas
import datetime

def _site_read_orig(sid, data_type, data_class, data_source, event, logger):

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
        time_p = parse_datetime(row[1])
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

  from swerve import config, parse_datetime
  CONFIG = config()
  data_dir = CONFIG['dirs']['original']

  data = []
  time = []

  if data_type == 'GIC' and data_class == 'measured' and data_source == 'TVA':
    data_dir = os.path.join(data_dir, 'tva', event, 'gic', 'GIC-measured')
    sid = sid.lower().replace(' ','')
    if event == '2024-05-10' or event == None:
      if sid == 'widowscreek':
        sid = f'{sid}2'
    fname = f'gic-{sid}_{event.replace("-", "")}.csv'
    file = os.path.join(data_dir, fname)
    logger.info(f"    Reading {file}")
    if not os.path.exists(file):
      fname = f'{sid}_event_{event.replace("-", "")[:6]}.csv'
      file = os.path.join(data_dir, fname)
      if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    with open(file, 'r') as csvfile:
      rows = csv.reader(csvfile, delimiter=',')
      for row in rows:
          if row[0] == 'Timestamp' or row[0] == 'timestamp': #skip header row if applicable
            continue
          time.append(parse_datetime(row[0]))
          data.append(float(row[1]) if row[1] != '' else numpy.nan)

    # Reshape to 2D array with a single column
    data = numpy.array(data).reshape(-1, 1)
    return {
      "time": numpy.array(time).flatten(),
      "data": data,
      "labels": ["GIC"],
      "unit": "A",
    }

  if data_type == 'GIC' and data_class == 'measured' and data_source == 'NERC':
    data_path = os.path.join(data_dir, data_source.lower(), event, 'gic')
    data_file = next((f for f in os.listdir(data_path) if f.endswith(f'{sid}.csv')), None)
    data = read_nerc(data_path, data_file)
    return {**data, "labels": ["GIC"], "unit": "A"}

  if data_type == 'GIC' and data_class == 'calculated' and data_source == 'TVA':
    data_dir = os.path.join(data_dir, 'tva', event, 'gic', 'GIC-calculated')
    sid = sid.replace(' ','')
    if sid == 'BullRun':
      sid = 'BullRunXfrm' # BullRun file Xfrm appended to name in file name.
    if sid == 'WidowsCreek':
      sid = f'{sid}2'

    dates = ['20240510', '20240511', '20240512'] #TODO: for 2024-05-10 storm ONLY
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
      fname = os.path.join(data_dir, 'gmu', event, 'nerc', f'site_{nearest_sim_site}.csv')
    elif 'TVA' in measured_sources:
      fname = os.path.join(data_dir, 'gmu', event, 'tva', f'site_{nearest_sim_site}.csv')
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
        time.append(parse_datetime(row[0]))
        #data.append([float(row[2]), float(row[3]), float(row[4])])
        data.append([float(row[2])])

    data = numpy.array(data)
    time = numpy.array(time)
    return {"time": time, "data": data, "labels": ["GIC"], "unit": "A"}

  if data_type == 'B' and data_class == 'measured' and data_source == 'TVA':
    data_dir = os.path.join(data_dir, 'tva', event, 'mag')
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
        time.append(parse_datetime(row[0]))
        data.append([float(row[1]), float(row[2]), float(row[3])])

    data = numpy.array(data)
    time = numpy.array(time)
    return {"time": time, "data": data, "labels": ["Bx", "By", "Bz"], "unit": "nT"}

  if data_type == 'B' and data_class == 'measured' and data_source == 'NERC':
    # TODO: magnetometers.csv indicates if GEO or MAG coordinates
    data_path = os.path.join(data_dir, data_source.lower(), event, 'mag')
    data_file = next((f for f in os.listdir(data_path) if f.endswith(f'{sid}.csv')), None)
    data = read_nerc(data_path, data_file)
    return {**data, "labels": ["B_N", "B_E", "B_v"], "unit": "nT"}

  if data_type == 'B' and data_class == 'calculated' and data_source in ['SWMF', 'OpenGGCM']:

    sid = sid.replace(' ','')
    data_dir = os.path.join(data_dir, data_source.lower(), event, sid.lower())

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
  
  if data_type == 'GIC' and data_class == 'measured' and data_source == 'Parry2025':
    from matio import load_from_mat
    data_file = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), f'{event.replace("-", "")}_hallprobe_data.mat')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    logger.info(f"    Reading {data_file}")
    load_data = load_from_mat(data_file)

    time = pandas.to_datetime(load_data['time_GIC']).to_pydatetime()
    if sid.lower().replace(' ','') == 'alberta1':
      data = load_data['Sub1_GIC']
    elif sid.lower().replace(' ','') == 'alberta2':
      data = load_data['Sub2_GIC']

    data = numpy.array(data).reshape(-1, 1)
    return {
      "time": numpy.array(time).flatten(),
      "data": data,
      "labels": ["GIC"],
      "unit": "A"
    }

  if data_type == 'B' and data_class == 'measured' and data_source == 'Parry2025':
    data_file = os.path.join(data_dir, data_source.lower(), event, 'mag', f'{event.replace('-','')}{sid.upper()}.F01')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    logger.info(f"    Reading {data_file}")
    time = []
    data = []
    with open(data_file, 'r', encoding="utf-8") as f:
      rows = f.readlines()
      for row in rows:
        if row.startswith(f'{sid.upper()}'):
            continue
        split_row = row.split()
        time.append(datetime.datetime.strptime(split_row[0], '%Y%m%d%H%M%S'))
        data_bx = float(split_row[1])
        data_by = float(split_row[2])
        data_bz = float(split_row[3])
        data.append([data_bx, data_by, data_bz])

    return {
              "time": numpy.array(time),
              "data": numpy.array(data),
              "labels": ["Bx", "By", "Bz"],
              "unit": "nT"
            }

  if data_type == 'GIC' and data_class == 'measured' and data_source == 'Parry2024':
      from datetime import timedelta, timezone
      from zoneinfo import ZoneInfo
      data_file = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), '20211012_GIC_data_89S.csv')
      if not os.path.exists(data_file):
          raise FileNotFoundError(f"Data file not found: {data_file}")
  
      data = []
      time = []

      if sid.lower().replace(' ','') == 'ellerslie1':
        data_col = 1
      elif sid.lower().replace(' ','') == 'ellerslie2':
        data_col = 2

      logger.info(f"    Reading {data_file}")

      with open(data_file, 'r') as csvfile:
        next(csvfile)  # Skip header rows
        next(csvfile)
        rows = csv.reader(csvfile, delimiter=',')
        for row in rows:
          timestamp = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M')
          # Time is in MDT (GMD-6), convert to UTC then make timezone naive for consistency
          timestamp_utc = timestamp.replace(tzinfo=ZoneInfo("America/Denver")).astimezone(timezone.utc)
          timestamp_utc = timestamp_utc.replace(tzinfo=None)
          time.append(timestamp_utc)
          data.append(float(row[data_col]) if row[data_col] != '#VALUE!' else numpy.nan)

      # Edit time column to match 0.5Hz measurement frequency described in paper
      corrected_time = [
        time[0] + timedelta(seconds=i * 2) 
        for i in range(len(time))
      ]

      data = numpy.array(data).reshape(-1, 1)
      return {
        "time": numpy.array(corrected_time).flatten(),
        "data": data,
        "labels": ["GIC"],
        "unit": "A"
      }

  if data_type == 'DMM' and data_class == 'measured' and data_source == 'Parry2024':
        from datetime import timedelta
        if sid.lower().replace(' ','') == 'albertaline':
          fname = f'{event.replace("-", "")}USB4.1Hz'
        elif sid.lower().replace(' ','') == 'albertaref':
          fname = f'{event.replace("-", "")}USB5.1Hz'
        data_file = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), fname)
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        data  = []
        time = []
        logger.info(f"    Reading {data_file}")
        with open(data_file, 'r') as csvfile:
              rows = csv.reader(csvfile, delimiter=',')
              for row in rows:
                  split_row = row[0].split()
                  time.append(datetime.datetime.strptime(split_row[0], '%Y%m%d%H%M%S'))
                  data_bx = float(split_row[1]) if split_row[1] != '' else numpy.nan
                  data_by = float(split_row[2]) if split_row[2] != '' else numpy.nan
                  data_bz = float(split_row[3]) if split_row[3] != '' else numpy.nan
                  data.append([data_bx, data_by, data_bz])
        return {
          "time": numpy.array(time),
          "data": numpy.array(data),
          "labels": ["Bx", "By", "Bz"],
          "unit": "nT"
        }

  if data_type == 'DMM' and data_class == 'measured' and (data_source == 'Marsal2025' or data_source == 'Marsal2021'):
      sid_name,sid_type = sid.strip().split()
      if data_source == 'Marsal2025':
        if sid_type == 'line':
          data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), sid_name, f'{sid_name}_LIN')
        elif sid_type == 'ref':
          data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), sid_name, f'{sid_name}_REF')
      if data_source == 'Marsal2021':
        if sid_type == 'line':
          data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), f'{sid_name}_lin')
        elif sid_type == 'ref':
          data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), f'{sid_name}_ref')

      if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

      time = []
      data =[]

      for item in os.listdir(data_path):
        logger.info(f"    Reading {os.path.join(data_path, item)}")
        with open(os.path.join(data_path, item), "r") as file:
            for line in file:
                if line.startswith('\x1a'):
                  continue
                # Split line by whitespace and convert to float
                row = line.split()
                timestamp = f'{row[0]}-{row[1]}-{row[2]}{row[3]}:{row[4]}:{row[5]}'
                time.append(datetime.datetime.strptime(timestamp, "%Y-%m-%d%H:%M:%S"))
                data_bx = float(row[6])
                data_by = float(row[7])
                data_bz = float(row[8])
                data.append([data_bx, data_by, data_bz])

      if len(time) != len(numpy.unique(time)):
        return {
                  "time": numpy.array(time),
                  "data": numpy.array(data),
                  "labels": ["Bx", "By", "Bz"],
                  "unit": "nT",
                  "error": 'Duplicate time stamps found'
                }
      else:
        return {
          "time": numpy.array(time),
          "data": numpy.array(data),
          "labels": ["Bx", "By", "Bz"],
          "unit": "nT"
        }
  
  if data_type == 'GIC' and data_class == 'measured' and data_source == 'Zhang2020':
    data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower())
    data_file = next((f for f in os.listdir(data_path) if (f.startswith(f'{event.replace('-','')}') and f.endswith('.txt'))), None)
    data_path = os.path.join(data_path, data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"    Reading {data_path}")
    time = []
    data = []

    with open(os.path.join(data_path), "r") as file:
      for line in file:
          # Remove header lines
          if line.startswith(':') or line.startswith('#'):
              continue
          # Remove empty lines
          if not line.strip():
              continue
          # Split line by whitespace and convert to float
          row = line.split()
          time.append(datetime.datetime.strptime(f'{row[0]}-{row[1]}-{row[2]} {row[3]}', "%Y-%m-%d %H%M%S"))
          data.append(float(row[4]) if row[4] != '' else numpy.nan)

    return {
            "time": numpy.array(time).flatten(),
            "data": numpy.array(data).reshape(-1, 1),
            "labels": ["GIC"],
            "unit": "A"
          }

  if data_type == 'GIC' and data_source == 'AlvesRibeiro':
    data_file = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), 'GIC_estimated_and_observed_storm_sep_2021.txt')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    logger.info(f"    Reading {data_file}")

    time = []
    data = []

    with open(data_file, "r") as file:
        next(file) # Skip the header line
        for line in file:
            row = line.split()
            time.append(datetime.datetime.strptime(f'{row[0]} {row[1]}', "%d/%m/%Y %H:%M"))
            if data_class == 'calculated':
              data.append(float(row[2]) if row[2] != '' else numpy.nan)
            if data_class == 'measured':
              data.append(float(row[3]) if row[3] != '' else numpy.nan)

    return {
            "time": numpy.array(time).flatten(),
            "data": numpy.array(data).reshape(-1, 1),
            "labels": ["GIC"],
            "unit": "A"
          }

  if data_type == 'GIC' and data_source == "Blake":
    data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower())
    data_file = next((f for f in os.listdir(data_path) if (f.startswith('gic_dataset_') and f.endswith('.txt'))), None)
    data_path = os.path.join(data_path, data_file)
    if not os.path.exists(data_path):
      raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"    Reading {data_path}")

    data = []

    with open(data_path, "r") as file:
      for line in file:
        # Remove header lines
        if line.startswith('M'):
            continue
        # Split line by whitespace and convert to float
        row = line.split()
        if data_class == 'measured':
          data.append(float(row[0]))
        if data_class == 'calculated':
          data.append(float(row[1]))

    start_time = datetime.datetime.strptime(f'{event}', "%Y-%m-%d")
    cadence = datetime.timedelta(minutes=1) # given data cadence
    time = [start_time + (i * cadence) for i in range(len(data))]

    return {
            "time": numpy.array(time).flatten(),
            "data": numpy.array(data).reshape(-1, 1),
            "labels": ["GIC"],
            "unit": "A"
          }

  if data_type == 'GIC' and data_class == 'measured' and data_source == 'Espinosa':
    import openpyxl
    data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower())
    data_file = os.path.join(data_path, 'swe20818-sup-0002-2018sw002094-table_si-s01.xlsx')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    logger.info(f"    Reading {data_file}")

    time = []
    data = []

    workbook = openpyxl.load_workbook(data_file)
    sheet = workbook.active
    for row in sheet.iter_rows(values_only=True):
        # Skip header row
        if row[1].startswith('Time'):
            continue
        time.append(datetime.datetime.strptime(f'{row[1]}', "%d-%b-%Y %H:%M:%S"))
        data.append(float(row[2]))
    
    return {
                "time": numpy.array(time).flatten(),
                "data": numpy.array(data).reshape(-1, 1),
                "labels": ["GIC"],
                "unit": "A"
              }

  if data_type == 'GIC' and data_class == 'measured' and data_source == 'Bailey':
    data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), 'gic_1and5_meas_sept2017.csv')
    if not os.path.exists(data_path):
      raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"    Reading {data_path}")

    time = []
    data = []

    with open(data_path, 'r') as csvfile:
      next(csvfile)  # Skip header row
      rows = csv.reader(csvfile, delimiter=',')
      for row in rows:
        time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
        if sid == 'Austria SS1':
          data.append(float(row[1]))
        if sid == 'Austria SS5':
          data.append(float(row[2]))
    
    return {
            "time": numpy.array(time).flatten(),
            "data": numpy.array(data).reshape(-1, 1),
            "labels": ["GIC"],
            "unit": "A"
          }

  if data_type == 'GIC' and data_class == 'calculated' and data_source == 'Bailey':
      data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), f'gic_{sid[-1]}_pred_sept2017.csv')
      if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
  
      logger.info(f"    Reading {data_path}")
  
      time = []
      data = []
  
      with open(data_path, 'r') as csvfile:
        next(csvfile)  # Skip header row
        rows = csv.reader(csvfile, delimiter=',')
        for row in rows:
          time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
          data.append(float(row[2]))
      
      return {
              "time": numpy.array(time).flatten(),
              "data": numpy.array(data).reshape(-1, 1),
              "labels": ["GIC"],
              "unit": "A"
            }

  if data_type == 'GIC' and data_class == 'measured' and data_source == 'Nahayo':
    if event == '2003-10-29':
      data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), 'nahayo_etal_data_event1.csv')
    if event == '2015-03-17':
      data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower(), 'nahayo_etal_data_event2.csv')
    if not os.path.exists(data_path):
      raise FileNotFoundError(f"Data file not found: {data_path}")
    

    logger.info(f"    Reading {data_path}")

    time = []
    data = []

    with open(data_path, 'r') as csvfile:
      rows = csv.reader(csvfile, delimiter=',')
      for row in rows:
        # Skip header rows
        if row[0].startswith('#'):
          continue
        time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
        data.append(float(row[8]))

    return {
            "time": numpy.array(time).flatten(),
            "data": numpy.array(data).reshape(-1, 1),
            "labels": ["GIC"],
            "unit": "A"
          }

  if data_type == 'GIC' and data_class == 'measured' and data_source == 'Watari':
    from datetime import timezone
    from zoneinfo import ZoneInfo
    data_path = os.path.join(data_dir, data_source.lower(), event, data_type.lower())
    if not os.path.exists(data_path):
      raise FileNotFoundError(f"Data dir not found: {data_path}")

    data = []

    for item in os.listdir(data_path):
      logger.info(f"    Reading {os.path.join(data_path, item)}")
      with open(os.path.join(data_path, item), "r") as file:
        for line in file:
          if line.startswith('\x1a'):
            continue
          row = line.split(',')
          data.append(float(row[0]))

    # Creating time array and converting from JST to UTC
    time = []
    start_time = datetime.datetime.strptime(f'{event}', "%Y-%m-%d")
    cadence = datetime.timedelta(seconds=1) # given data cadence
    time_jst = [start_time + (i * cadence) for i in range(len(data))]
    for timestamp in time_jst:
      timestamp_utc = timestamp.replace(tzinfo=ZoneInfo("Asia/Tokyo")).astimezone(timezone.utc)
      timestamp_utc = timestamp_utc.replace(tzinfo=None)
      time.append(timestamp_utc)

    return {
            "time": numpy.array(time).flatten(),
            "data": numpy.array(data).reshape(-1, 1),
            "labels": ["GIC"],
            "unit": "A"
          }

  if data_type == 'B' and data_class == 'measured' and data_source == 'Watari':
    data_path = os.path.join(data_dir, data_source.lower(), event, 'mag')
    if not os.path.exists(data_path):
      raise FileNotFoundError(f"Data dir not found: {data_path}")

    time = []
    data = []

    for item in os.listdir(data_path):
      logger.info(f"    Reading {os.path.join(data_path, item)}")
      with open(os.path.join(data_path, item), "r") as file:
        for line in file:
          if line.startswith(' ') or line.startswith('DATE'):
            continue
          row = line.split()
          time.append(datetime.datetime.strptime(f'{row[0]} {row[1]}', '%Y-%m-%d %H:%M:%S.%f'))
          data.append([float(row[3]), float(row[4]), float(row[5])])

    return {
            "time": numpy.array(time),
            "data": numpy.array(data),
            "labels": ["Bx", "By", "Bz"],
            "unit": "nT"
          }

    


