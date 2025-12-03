import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from datetime import datetime, timedelta
from datetick import datetick

from swerve import config, plt_config, savefig, savefig_paper, subset

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

plot_both = True # if True, plot data used for MAGE and SWMF/OpenGGCM
plot_kp_compare = True # if True, compare Kp from MAGE and OMNI2

def read_mage(limits):
  import os
  import pickle

  mage_bcwind_h5 = CONFIG['files']['mage']['bcwind'] 
  mage_bcwind_pkl = mage_bcwind_h5.replace('.h5', '.pkl')
  if os.path.exists(mage_bcwind_pkl):
    logger.info(f'Reading cached data from {mage_bcwind_pkl}')
    with open(mage_bcwind_pkl, 'rb') as f:
      return pickle.load(f)

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
  columns = ['year', 'month', 'day', 'hour', 'min', 'sec', 'msec', 'Bx[nT]', 'By[nT]', 'Bz[nT]', 'Vx[km/s]', 'Vy[km/s]', 'Vz[km/s]', 'N[cm^(-3)]', 'T[Kelvin]']
  data = pd.read_csv(CONFIG['files']['swmf']['bcwind'], delim_whitespace=True, names=columns, header=0)
  time = np.array([
      datetime(row.year, row.month, row.day, row.hour, row.min, row.sec) +
      timedelta(seconds=round(row.msec / 1000))
      for row in data.itertuples()
  ])

  data['time'] = time
  df = data[['time', 'Bx[nT]', 'By[nT]', 'Bz[nT]', 'Vx[km/s]', 'Vy[km/s]', 'Vz[km/s]', 'N[cm^(-3)]', 'T[Kelvin]']]
  df = df.rename(columns={
      'Bx[nT]': 'Bx',
      'By[nT]': 'By',
      'Bz[nT]': 'Bz',
      'Vx[km/s]': 'Vx',
      'Vy[km/s]': 'Vy',
      'Vz[km/s]': 'Vz',
      'N[cm^(-3)]': 'N',
      'T[Kelvin]': 'T'
  })

  df = df[(df['time'] >= limits[0]) & (df['time'] <= limits[1])]

  return df

def read_kp_omni2():
  # View data using
  # https://hapi-server.org/servers/#server=CDAWeb&dataset=OMNI2_H0_MRG1HR&parameters=KP1800&start=2024-05-10&stop=2024-05-13&return=data&format=csv&style=noheader
  from hapiclient import hapi, hapitime2datetime

  server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
  dataset    = 'OMNI2_H0_MRG1HR'
  parameters = 'KP1800'
  start      = '2024-05-10' # min 1963-01-01T00:00:00Z
  stop       = '2024-05-12' # max 2025-06-12T17:00:00Z

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
  savefig('data_processed\summary\_imf', 'mage_omni2_kp_compare', logger)
  plt.close()

limits_data = CONFIG['limits']['data']
limits_plot = CONFIG['limits']['plot']

data = read_mage(limits_data)

if plot_kp_compare:
  time, kp, timex, kpx = read_kp_omni2()
  compare_kp(time, kp, data['time'], data['Kp'])

fig = plt.figure(figsize=(8.5, 11))
gs = plt.gcf().add_gridspec(7, 1, hspace=0.15)
axes = gs.subplots(sharex=True)

# Plotting al and ae
axes[0].plot(data['time'], -data['al'], label=r'$-$SME L', color='k', linewidth=1)
axes[0].plot(data['time'], data['ae'], label='SME U', color='m', linewidth=0.5)
axes[0].set_ylabel('[nT]')
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
axes[2].plot(data['time'], data['symh'], color='k', linewidth=0.8)
axes[2].set_ylabel('SYM-H [nT]')

# Plotting temperature
axes[3].plot(data['time'], data['Temp'] / 1e6, color='k', linewidth=0.8, label='T (MAGE)')
axes[3].set_ylabel(r'T [MK]')
axes[3].yaxis.set_major_locator(MultipleLocator(2))

# Plotting mach
axes[4].plot(data['time'], data['Magnetosonic Mach'], color='k', linewidth=0.8)
axes[4].set_ylabel("Mag Mach")
axes[4].yaxis.set_major_locator(MultipleLocator(0.01))
axes[4].set_ylim(0, 0.03)

# Plotting Vx
# divide by 1000 to get in km/s
axes[5].plot(data['time'], data['Vx']/1000, color='k', linewidth=0.8, label=r'V$_x$ (MAGE)')
axes[5].set_ylabel(r'V$_x$ [km/s]')
axes[5].yaxis.set_major_locator(MultipleLocator(200))

# Plotting Bx, By, Bz IMF
axes[6].plot(data['time'], data['By'], label=r'B$_y^\text{IMF}$', color='k', linewidth=0.5)
axes[6].plot(data['time'], data['Bz'], label=r'B$_z^\text{IMF}$', color='m', linewidth=0.5)
axes[6].set_ylabel('[nT]')
axes[6].legend(loc='lower right', ncol=2, frameon=False)
axes[6].yaxis.set_major_locator(MultipleLocator(50))

plt_adjust(limits_plot)
fig.align_ylabels(axes)

savefig('data_processed\summary\_imf', 'imf_mage', logger)
savefig_paper('_imf', 'imf_mage', logger)

if plot_both:
  df = read_swmf(limits_data)
  ## To plot Magnetosonic Mach, see:
  ##  https://omniweb.gsfc.nasa.gov/ftpbrowser/bow_derivation.html
  ##  https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2009JA014998

  # Adding T to existing axes
  axes[3].plot(df['time'], df['T']/1e6, label='T (SWMF)', color='gray', linewidth=0.8)
  axes[3].legend(loc='upper right', ncol=2)

  # Adding Vx to existing axes
  axes[5].plot(df['time'], df['Vx'], label=r'V$_x$ (SWMF)', color='gray', linewidth=0.8)
  axes[5].legend(loc='upper right', ncol=2)

  # Adding By and Bz to the existing axes
  axes[6].plot(df['time'], df['By'], label=r'B$_y^\text{SWMF}$', color='gray', linewidth=0.8)
  axes[6].plot(df['time'], df['Bz'], label=r'B$_z^\text{SWMF}$', color='dimgray', linewidth=0.5, linestyle='--')
  axes[6].legend(loc='upper right', ncol=4)

  savefig('data_processed\summary\_imf', 'imf_all', logger)
