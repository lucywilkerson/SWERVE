import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from datetime import datetime, timedelta
from datetick import datetick

from swerve import cli, config, plt_config, savefig, savefig_paper, subset

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
plt_config()

plot_all = True # if True, plot data used for MAGE and SWMF/OpenGGCM and raw data from OMNI
plot_kp_compare = False # if True, compare Kp from MAGE and OMNI2
plot_hapi = False  # if True, make IMF plot using HAPI data

args = cli('main.py')
paper = False
if args['event'] == '2024-05-10':
  paper = True

def read_mage(limits):
  import os
  import pickle

  mage_bcwind_h5 = CONFIG['files']['mage']['bcwind'] 
  mage_bcwind_pkl = mage_bcwind_h5.replace('.h5', '.pkl')
  if os.path.exists(mage_bcwind_pkl):
    logger.info(f'Reading cached data from {mage_bcwind_pkl}')
    with open(mage_bcwind_pkl, 'rb') as f:
      return pickle.load(f)
  else:
    logger.info(f'Cached file {mage_bcwind_pkl} not found. Obtaining from HAPI server.')
    plot_hapi = True
    return None

  logger.info(f'Reading {mage_bcwind_h5}')
  h5file = h5py.File(mage_bcwind_h5,'r')

  time = np.array([datetime(1858, 11, 17) + timedelta(days=day) for day in h5file['MJD'][()]])
  data = {}
  for key in h5file.keys():
    data[key] = h5file[key][()]
    logger.info(f'  {key}.shape = {data[key].shape}')
    if isinstance(data[key], np.ndarray) and len(data[key]) == len(time):
      data['time'], data[key] = subset(time, data[key], limits[0], limits[1])

  with open(mage_bcwind_pkl, 'wb') as f:
    pickle.dump(data, f)

  return data

def read_swmf(limits):
  # Reading Dean's data used for SWMF and OpenGGCM runs at CCMC
  logger.info(f'Reading {CONFIG['files']['swmf']['bcwind']}')
  columns = ['year', 'month', 'day', 'hour', 'min', 'sec', 'msec', 'Bx(nT)', 'By(nT)', 'Bz(nT)', 'Vx(km/s)', 'Vy(km/s)', 'Vz(km/s)', 'N(cm^(-3))', 'T(Kelvin)']
  data = pd.read_csv(CONFIG['files']['swmf']['bcwind'], delim_whitespace=True, names=columns, header=0)
  time = np.array([
      datetime(row.year, row.month, row.day, row.hour, row.min, row.sec) +
      timedelta(seconds=round(row.msec / 1000))
      for row in data.itertuples()
  ])

  data['time'] = time
  df = data[['time', 'Bx(nT)', 'By(nT)', 'Bz(nT)', 'Vx(km/s)', 'Vy(km/s)', 'Vz(km/s)', 'N(cm^(-3))', 'T(Kelvin)']]
  df = df.rename(columns={
      'Bx(nT)': 'Bx',
      'By(nT)': 'By',
      'Bz(nT)': 'Bz',
      'Vx(km/s)': 'Vx',
      'Vy(km/s)': 'Vy',
      'Vz(km/s)': 'Vz',
      'N(cm^(-3))': 'N',
      'T(Kelvin)': 'T'
  })
  df = df.sort_values('time').reset_index(drop=True)

  df = df[(df['time'] >= limits[0]) & (df['time'] <= limits[1])]

  return df

def read_kp_omni2(start='2024-05-10', stop='2024-05-12'):
  # View data using
  # https://hapi-server.org/servers/#server=CDAWeb&dataset=OMNI2_H0_MRG1HR&parameters=KP1800&start=2024-05-10&stop=2024-05-13&return=data&format=csv&style=noheader
  from hapiclient import hapi, hapitime2datetime

  server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
  dataset    = 'OMNI2_H0_MRG1HR'
  parameters = 'KP1800'
  start      = start # default 1963-01-01T00:00:00Z
  stop       = stop # default max 2025-06-12T17:00:00Z

  data, meta = hapi(server, dataset, parameters, start, stop)
  time = hapitime2datetime(data['Time'])
  kp = data['KP1800']/10

  kpx = []
  timex = []
  for i, time in enumerate(timex):
    # Select Kp values at 02:30, 05:30, ...
    # Set timestamp to be 03:00, 06:00, ... for use with step
    if time.hour in np.arange(2, 24, 3) and time.minute == 30:
      kpx.append(data_kp[i])
      timex.append(time + timedelta(minutes=30))

  return time, kp, timex, kpx

def read_hapi(limits_plot):
  #TODO: save file locally with start and stop in name (pkl)
  from hapiclient import hapi, hapitime2datetime
  import pandas

  server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
  dataset    = 'OMNI2_H0_MRG1HR'  
  parameters = 'T1800,Mgs_mach_num1800,DST1800,AE1800,AL_INDEX1800,AU_INDEX1800'
  start      = limits_plot[0].strftime('%Y-%m-%dT%H:%M:%SZ')
  stop       = limits_plot[1].strftime('%Y-%m-%dT%H:%M:%SZ')
  data_hapi, meta = hapi(server, dataset, parameters, start, stop, logging=True)

  parameters_dict = {
    'Time': 'Time',
    'T1800': 'Temp',
    'Mgs_mach_num1800': 'Magnetosonic Mach',
    'DST1800': 'Dst',
    'AE1800': 'AE',
    'AL_INDEX1800': 'AL',
    'AU_INDEX1800': 'AU'
  }
  
  dfs = []
  for param, name in parameters_dict.items():
    if param == 'Time':
      dfs.append(pandas.DataFrame(hapitime2datetime(data_hapi[param])))
    else:
      dfs.append(pandas.DataFrame(data_hapi[param]))
  
  df_1hr = pandas.concat(dfs, axis=1)
  df_1hr.columns = list(parameters_dict.values())

  dataset    = 'OMNI_HRO2_1MIN'
  parameters = 'BX_GSE,BY_GSE,BZ_GSE,Vx,T,Mgs_mach_num,AE_INDEX,AL_INDEX,AU_INDEX,SYM_H'
  data_hapi, meta = hapi(server, dataset, parameters, start, stop)

  parameters_dict = {
    'Time': 'Time',
    'BX_GSE': 'Bx',
    'BY_GSE': 'By',
    'BZ_GSE': 'Bz',
    'Vx': 'Vx',
    'T': 'Temp',
    'Mgs_mach_num': 'Magnetosonic Mach',
    'SYM_H': 'SYM_H',
    'AE_INDEX': 'AE',
    'AL_INDEX': 'AL',
    'AU_INDEX': 'AU'
  }
  
  dfs = []
  i=0
  for param, name in parameters_dict.items():
    if param == 'Time':
      dfs.append(pandas.DataFrame(hapitime2datetime(data_hapi[param])))
    else:
      dfs.append(pandas.DataFrame(data_hapi[param]))
      fill = meta['parameters'][i]['fill']
      if fill is not None:
        fill = float(fill)
        dfs[-1][dfs[-1] == fill] = np.nan
    i+=1
  df_1m = pandas.concat(dfs, axis=1)
  df_1m.columns = list(parameters_dict.values())
  
  return df_1hr, df_1m

def plt_adjust(xlims):
  for ax in axes:
    ax.set_xlim(xlims)
    ax.grid(True)
    datetick('x', axes=ax)
    if ax != axes[1]:
      ax.minorticks_on()
      ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.5)

    leg = ax.get_legend()
    if leg is not None:
      # change the line width for the legend
      for line in leg.get_lines():
        line.set_linewidth(2)

  # Remove x-axis labels for all subplots except the bottom one
  for ax in axes[:-1]:
    ax.set_xticklabels([])
  datetick('x', axes=axes[-1])

def compare_kp(time_omni2, kp_omni2, time_mage, data_mage):
  # MAGE data is on one-second time grid. Kp appears to be interpolated to that grid.
  # OMNI2 timestamps are at 00:30, 01:30, ...
  # These don't match and transitions in MAGE don't occur at 00:00, 03:00, etc.
  plt.plot(time_omni2, kp_omni2, 'b.', label='OMNI2_H0_MRG1HR')
  plt.plot(time_mage, data_mage, 'k.', label='MAGE')
  plt.ylabel(r'K$_p$')
  plt.grid()
  datetick('x')
  plt.legend()
  savefig('data_processed/summary/_imf', 'mage_omni2_kp_compare', logger)
  plt.close()

def mage_swmf_rmse(mage_values, swmf_values, parameter, start_time, stop_time):
    import pandas as pd

    def filter_by_mage_and_swmf(mage_values, swmf_values, parameter, start_time, stop_time):
      # Create masks for each dataset independently
      mage_mask = (mage_values['time'] >= start_time) & (mage_values['time'] <= stop_time)
      swmf_mask = (swmf_values['time'] >= start_time) & (swmf_values['time'] <= stop_time)

      # Filter the specific parameter data using the masks
      filtered_mage = mage_values[parameter][mage_mask]
      filtered_swmf = swmf_values[parameter][swmf_mask]

      # Interpolate SWMF data to match MAGE time grid
      mage_time = pd.Series(mage_values['time'][mage_mask])
      swmf_values_temp = swmf_values.drop_duplicates(subset=['time']).set_index('time').reindex(mage_time).interpolate().reset_index()
      swmf_values_temp.columns = ['time'] + list(swmf_values_temp.columns[1:])
      filtered_mage = mage_values[parameter][mage_mask]
      filtered_swmf = swmf_values_temp[parameter]
      
      # Return the filtered times and the data
      return (mage_values['time'][mage_mask], filtered_mage, 
              swmf_values_temp['time'], filtered_swmf)
    
    _, mage_filtered, _, swmf_filtered = filter_by_mage_and_swmf(mage_values, swmf_values, parameter, start_time, stop_time)
    rmse = np.sqrt(np.mean((mage_filtered - swmf_filtered)**2))
    if parameter in ['Vx']:
      unit = 'km/s'
    elif parameter in ['By', 'Bz']:
      unit = 'nT'
    else:
      unit = ''
    print(f'RMSE between MAGE and SWMF {parameter}: {rmse:.2f} {unit}')
    return rmse

limits_data = CONFIG['limits']['data']
limits_plot = CONFIG['limits']['plot']

data = read_mage(limits_data)

if plot_hapi:
    start = limits_plot[0].strftime('%Y-%m-%dT%H:%M:%SZ')
    stop = limits_plot[1].strftime('%Y-%m-%dT%H:%M:%SZ')

    df_1hr, df_1m = read_hapi(limits_plot)
    
    fig = plt.figure(figsize=(8.5, 11))
    gs = plt.gcf().add_gridspec(7, 1, hspace=0.15)
    axes = gs.subplots(sharex=True)

    # Plotting -al and ae
    axes[0].plot(df_1m['Time'], -df_1m['AL'], label=r'$-$AL', color='k', linewidth=1)
    axes[0].plot(df_1m['Time'], df_1m['AE'], label='AE', color='m', linewidth=0.5)
    axes[0].set_ylabel('(nT)')
    axes[0].yaxis.set_major_locator(MultipleLocator(2000))
    axes[0].legend(ncol=2, frameon=False)

    # Plotting Kp
    time, kp, timex, kpx = read_kp_omni2(start=start, stop=stop)
    axes[1].step(time, kp, color='k', where='post')
    axes[1].fill_between(time, kp, 0, step='post', color='k')
    axes[1].plot(timex, kpx, 'b.')
    axes[1].set_ylabel(r'K$_p$')
    axes[1].yaxis.set_major_locator(MultipleLocator(3))
    axes[1].set_ylim(0, 9.5)

    # Plotting symh and dst
    axes[2].plot(df_1m['Time'], df_1m['SYM_H'], color='k', linewidth=1, label='SYM-H')
    axes[2].plot(df_1hr['Time'], df_1hr['Dst'], color='m', linewidth=0.5, label='Dst')
    axes[2].set_ylabel('(nT)')
    axes[2].legend(ncol=2, frameon=False)

    # Plotting temperature
    axes[3].plot(df_1m['Time'], df_1m['Temp'] / 1e6, color='k', linewidth=0.8, label='T')
    axes[3].set_ylabel(r'T (MK)')
    axes[3].yaxis.set_major_locator(MultipleLocator(.5))

    # Plotting ms mach
    axes[4].plot(df_1m['Time'], df_1m['Magnetosonic Mach'], color='k', linewidth=0.8)
    axes[4].set_ylabel("Mag Mach")

    # Plotting Vx
    axes[5].plot(df_1m['Time'], df_1m['Vx'], color='k', linewidth=0.8, label=r'V$_x$ (HAPI)')
    axes[5].set_ylabel(r'V$_x$ (km/s)')
    axes[5].yaxis.set_major_locator(MultipleLocator(200))

    # Plotting By, Bz IMF
    axes[6].plot(df_1m['Time'], df_1m['By'], label=r'B$_y^\text{IMF}$', color='k', linewidth=0.5)
    axes[6].plot(df_1m['Time'], df_1m['Bz'], label=r'B$_z^\text{IMF}$', color='m', linewidth=0.5)
    axes[6].set_ylabel('(nT)')
    axes[6].legend(loc='lower right', ncol=2, frameon=False)
    axes[6].yaxis.set_major_locator(MultipleLocator(50))

    plt_adjust(limits_plot)
    fig.align_ylabels(axes)
    savefig('data_processed/summary/_imf', 'imf_hapi', logger)
    if paper:
      savefig_paper('figures/_imf', 'imf_hapi', logger)

    # add imf parameters to regression results
    import pickle
    import os
    fname = CONFIG['files']['regression_results']['gic_max']
    if os.path.exists(fname):
      with open(fname, 'rb') as f:
        fit_models = pickle.load(f)
    else:
      logger.error(f"File {fname} does not exist. Rerun regression.py with reparse=True to create it.")
    current_event = args['event']
    if current_event == None: current_event = '2024-05-10'
    if current_event in fit_models.keys():
      min_dst = df_1hr['Dst'].min()
      fit_models[current_event]['min_dst'] = min_dst
      min_symh = df_1m['SYM_H'].min()
      fit_models[current_event]['min_symh'] = min_symh
      min_bz = df_1m['Bz'].min()
      fit_models[current_event]['min_bz'] = min_bz
      max_kp = kp.max()
      fit_models[current_event]['max_kp'] = max_kp
      max_ae = df_1m['AE'].max()
      fit_models[current_event]['max_ae'] = max_ae
      max_vx = df_1m['Vx'].max()
      fit_models[current_event]['max_vx'] = max_vx
    if not os.path.exists(os.path.dirname(fname)):
      os.makedirs(os.path.dirname(fname))
    with open(fname, 'wb') as f:
      logger.info(f"Saving IMF parameters for {current_event} to {fname}")
      pickle.dump(fit_models, f)

if plot_kp_compare:
  time, kp, timex, kpx = read_kp_omni2()
  compare_kp(time, kp, data['time'], data['Kp'])

fig = plt.figure(figsize=(8.5, 11))
gs = plt.gcf().add_gridspec(7, 1, hspace=0.15)
axes = gs.subplots(sharex=True)

# Plotting al and ae
axes[0].plot(data['time'], -data['al'], label=r'$-$AL', color='k', linewidth=1)
axes[0].plot(data['time'], data['ae'], label='AE', color='m', linewidth=0.5)
axes[0].set_ylabel('(nT)')
axes[0].yaxis.set_major_locator(MultipleLocator(2000))
axes[0].legend(ncol=2, frameon=False)

# Plotting Kp
time, kp, timex, kpx = read_kp_omni2()
axes[1].step(time, kp, color='k', where='post')
axes[1].fill_between(time, kp, 0, step='post', color='k')
axes[1].plot(timex, kpx, 'b.')
axes[1].set_ylabel(r'K$_p$')
axes[1].yaxis.set_major_locator(MultipleLocator(3))
axes[1].set_ylim(0, 9.5)

# Plotting symh
axes[2].plot(data['time'], data['symh'], color='k', linewidth=0.8, label='SYM-H (MAGE)')
axes[2].set_ylabel('SYM-H (nT)')

# Plotting temperature
axes[3].plot(data['time'], data['Temp'] / 1e6, color='k', linewidth=0.8, label='T (MAGE)')
axes[3].set_ylabel(r'T (MK)')
axes[3].yaxis.set_major_locator(MultipleLocator(2))

# Plotting mach
axes[4].plot(data['time'], data['Magnetosonic Mach'], color='k', linewidth=0.8, label='Mag Mach (MAGE)')
axes[4].set_ylabel("Mag Mach")
axes[4].yaxis.set_major_locator(MultipleLocator(0.01))
axes[4].set_ylim(0, 0.03)

# Plotting Vx
# divide by 1000 to get in km/s
axes[5].plot(data['time'], data['Vx']/1000, color='k', linewidth=0.8, label=r'V$_x$ (MAGE)')
axes[5].set_ylabel(r'V$_x$ (km/s)')
axes[5].yaxis.set_major_locator(MultipleLocator(200))

# Plotting Bx, By, Bz IMF
axes[6].plot(data['time'], data['By'], label=r'B$_y^\text{IMF}$', color='k', linewidth=0.5)
axes[6].plot(data['time'], data['Bz'], label=r'B$_z^\text{IMF}$', color='m', linewidth=0.5)
axes[6].set_ylabel('(nT)')
axes[6].legend(loc='lower right', ncol=2, frameon=False)
axes[6].yaxis.set_major_locator(MultipleLocator(50))

plt_adjust(limits_plot)
fig.align_ylabels(axes)

savefig('data_processed/summary/_imf', 'imf_mage', logger)
savefig_paper('figures/_imf', 'imf_mage', logger)

if plot_all:
  df = read_swmf(limits_data)
  ## To plot Magnetosonic Mach, see:
  ##  https://omniweb.gsfc.nasa.gov/ftpbrowser/bow_derivation.html
  ##  https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2009JA014998

  if not plot_hapi: # only read HAPI data if we haven't already read it for the previous plot
    df_1hr, df_1m = read_hapi(limits_plot)

  # Adding -al and ae to existing axes
  axes[0].plot(df_1m['Time'], -df_1m['AL'], label=r'$-$AL (OMNI)', color='c', linewidth=0.8, linestyle=':')
  axes[0].plot(df_1m['Time'], df_1m['AE'], label='AE (OMNI)', color='b', linewidth=0.5, linestyle=':')
  axes[0].legend(loc='upper right', ncol=4)

  # Adding SYM-H and Dst to existing axes
  axes[2].plot(df_1m['Time'], df_1m['SYM_H'], color='c', linewidth=0.8, linestyle=':', label='SYM-H (OMNI)')
  axes[2].plot(df_1hr['Time'], df_1hr['Dst'], color='b', linewidth=0.5, linestyle=':', label='Dst (OMNI)')
  axes[2].legend(loc='upper right', ncol=3)

  # Adding T to existing axes
  axes[3].plot(df['time'], df['T']/1e6, label='T (SWMF)', color='gray', linewidth=0.8)
  axes[3].plot(df_1m['Time'], df_1m['Temp'] / 1e6, color='c', linewidth=0.8, linestyle=':', label='T (OMNI)')
  axes[3].legend(loc='upper right', ncol=3)

  # Adding Magnetosonic Mach to existing axes
  axes[4].plot(df_1m['Time'], df_1m['Magnetosonic Mach'], color='c', linewidth=0.8, linestyle=':', label='Mag Mach (OMNI)')
  axes[4].legend(loc='upper right', ncol=2)
  axes[4].yaxis.set_major_locator(MultipleLocator(5))
  axes[4].set_ylim(-1,18)
  
  # Adding Vx to existing axes
  axes[5].plot(df['time'], df['Vx'], label=r'V$_x$ (SWMF)', color='gray', linewidth=0.8)
  axes[5].plot(df_1m['Time'], df_1m['Vx'], color='c', linewidth=0.8, linestyle=':', label=r'V$_x$ (OMNI)')
  axes[5].legend(loc='upper right', ncol=3)
  # Adding RMSE calculation for Vx
  vx_rmse = mage_swmf_rmse(data, df, 'Vx', pd.to_datetime('2024-05-10T12:00:00'), pd.to_datetime('2024-05-10T12:00:00'))
  # Add RMSE to title
  axes[5].set_title(f'Vx Comparison (RMSE: {vx_rmse:.2f} km/s)', fontsize=10)

  # Adding By and Bz to the existing axes
  axes[6].plot(df['time'], df['By'], label=r'B$_y^\text{SWMF}$', color='orange', linewidth=0.8)
  axes[6].plot(df['time'], df['Bz'], label=r'B$_z^\text{SWMF}$', color='gray', linewidth=0.5, linestyle='--')
  axes[6].plot(df_1m['Time'], df_1m['By'], label=r'B$_y^\text{OMNI}$', color='c', linewidth=0.8, linestyle=':')
  axes[6].plot(df_1m['Time'], df_1m['Bz'], label=r'B$_z^\text{OMNI}$', color='b', linewidth=0.5, linestyle=':')
  axes[6].legend(loc='upper right', ncol=6)
  # Similar RMSE calcualtions for By and Bz
  by_rmse = mage_swmf_rmse(data, df, 'By', pd.to_datetime('2024-05-10T12:00:00'), pd.to_datetime('2024-05-10T12:00:00'))
  bz_rmse = mage_swmf_rmse(data, df, 'Bz', pd.to_datetime('2024-05-10T12:00:00'), pd.to_datetime('2024-05-10T12:00:00'))
  # Adding both RMSE to title
  axes[6].set_title(f'By Comparison (RMSE: {by_rmse:.2f} nT), Bz Comparison (RMSE: {bz_rmse:.2f} nT)', fontsize=10)

  savefig('data_processed/summary/_imf', 'imf_all', logger)
  plt.close()

  # New figure with just Vx, By, and Bz for better visibility
  fig = plt.figure(figsize=(8.5, 11))
  gs = plt.gcf().add_gridspec(3, 1, hspace=0.15)
  axes = gs.subplots(sharex=True)

  # Plotting Vx
  axes[0].plot(df_1m['Time'], df_1m['Vx'], color='k', linewidth=0.8, label=r'OMNI')
  axes[0].plot(data['time'], data['Vx']/1000, color='c', linestyle='-.', linewidth=0.8, label=r'MAGE')
  axes[0].plot(df['time'], df['Vx'], label=r'SWMF/GGCM', color='orange', linestyle=':', linewidth=0.8)
  axes[0].set_ylabel(r'V$_x$ (km/s)')
  axes[0].legend(loc='upper right', ncol=3)
  axes[0].yaxis.set_major_locator(MultipleLocator(200))

  # Plotting By
  axes[1].plot(df_1m['Time'], df_1m['By'], label=r'OMNI', color='k', linewidth=0.8)
  axes[1].plot(data['time'], data['By'], color='c', linestyle='-.', linewidth=0.8, label=r'MAGE')
  axes[1].plot(df['time'], df['By'], label=r'SWMF/GGCM', color='orange', linestyle=':', linewidth=0.8)
  axes[1].set_ylabel(r'B$_y^\text{IMF}$ (nT)')
  axes[1].legend(loc='upper right', ncol=3)
  axes[1].yaxis.set_major_locator(MultipleLocator(50))

  # Plotting Bz
  axes[2].plot(df_1m['Time'], df_1m['Bz'], label=r'OMNI', color='k', linewidth=0.8)
  axes[2].plot(data['time'], data['Bz'], color='c', linestyle='-.', linewidth=0.8, label=r'MAGE')
  axes[2].plot(df['time'], df['Bz'], label=r'SWMF/GGCM', color='orange', linestyle=':', linewidth=0.8)
  axes[2].set_ylabel(r'B$_z^\text{IMF}$ (nT)')
  axes[2].legend(loc='upper right', ncol=3)
  axes[2].yaxis.set_major_locator(MultipleLocator(50))

  plt_adjust(limits_plot)
  fig.align_ylabels(axes)
  
  savefig('data_processed/summary/_imf', 'imf_vx_by_bz', logger)
  savefig_paper('figures/_imf', 'imf_vx_by_bz', logger)
  
