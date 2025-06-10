import os
import numpy as np
import h5py
from datetime import datetime, timedelta

from datetick import datetick

from storminator import FILES, LOG_DIR, plt_config, savefig, savefig_paper, subset

import utilrsw
import pandas as pd
from matplotlib.ticker import MultipleLocator
logger = utilrsw.logger(log_dir=LOG_DIR)

import matplotlib.pyplot as plt

Both = False # if true, plot both MAGE and Dean's IMF
dean_fname = os.path.join('..', '2024-May-Storm-data', 'imf_data', 'Dean_IMF.txt')

def read(mage_bcwind_h5, limits):
  logger.info(f'Reading {mage_bcwind_h5}')
  h5file = h5py.File(mage_bcwind_h5,'r')

  time = np.array([datetime(1858, 11, 17) + timedelta(days=day) for day in h5file['MJD'][()]])
  data = {}
  for key in h5file.keys():
    logger.info(f'  Found {key}')
    data[key] = h5file[key][()]
    if isinstance(data[key], np.ndarray) and len(data[key]) ==  len(time):
      data['time'], data[key] = subset(time, data[key], limits['xlims'][0], limits['data'][1])

  for i in range(data['al'].size):
    if data['time'][i] <= datetime(2024, 5, 10, 0, 0):
      data['al'][i] = np.nan
      data['al'][i] = np.nan

  return data


def plt_adjust(xlims):
  xlims = [limits['xlims'][0], limits['xlims'][1]]
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
        line.set_linewidth(1)

  # Remove x-axis labels for all subplots except the bottom one
  for ax in axes[:-1]:
    ax.set_xticklabels([])
  datetick('x', axes=axes[-1])


limits = plt_config()
data = read(FILES['mage']['bcwind'], limits)

fig = plt.figure(figsize=(8.5, 11))
gs = plt.gcf().add_gridspec(7, 1)
axes = gs.subplots(sharex=True)

# Plotting al and ae
axes[0].plot(data['time'], -data['al'], label=r'$-$AL', color='k', linewidth=1)
axes[0].plot(data['time'], data['ae'], label='AE', color='m', linewidth=0.5)
axes[0].set_ylabel('[nT]')
axes[0].legend(ncol=2)

# Plotting Kp
# TODO: Should be able to do this without the loop (using only plt.stairs or plt.step).
kp_times = []
kp_values = []
# Creating a 1-hour grid from limits['data'][0] to limits['data'][1]
current_time = limits['xlims'][0]
while current_time <= limits['data'][1]:
  # Finding the index of the closest Kp value (Kp is every 3 hours, centered at :30)
  # Find the previous 3-hour interval
  kp_idx = None
  for i, t in enumerate(data['time']):
    if t.hour % 3 == 0 and t.minute == 0:
      kp_time_center = t + timedelta(hours=1.5)
      # If current_time falls within this 3-hour interval, use this Kp
      if kp_time_center - timedelta(hours=1.5) <= current_time < kp_time_center + timedelta(hours=1.5):
        kp_idx = i
        break
  if kp_idx is not None:
    kp_times.append(current_time)
    kp_values.append(data['Kp'][kp_idx])
  current_time += timedelta(hours=1)

axes[1].step(kp_times, kp_values, where='post', color='k')
axes[1].fill_between(kp_times, kp_values, 0, step='post', color='k')
axes[1].set_ylabel(r'K$_p$')
axes[1].yaxis.set_major_locator(MultipleLocator(3))
axes[1].set_ylim(0, 9.5)
#axes[1].set_ylim(4.5, 9.5)

# Plotting symh
axes[2].plot(data['time'], data['symh'], color='k', linewidth=0.8)
axes[2].set_ylabel('SYM-H [nT]')

# Plotting temperature
axes[3].plot(data['time'], data['Temp'] / 1e6, color='k', linewidth=0.8, label='T (MAGE)')
axes[3].set_ylabel(r'T [MK]')

# Plotting mach
axes[4].plot(data['time'], data['Magnetosonic Mach'], color='k', linewidth=0.8)
axes[4].set_ylabel("Mag Mach")

# Plotting Vx
axes[5].plot(data['time'], data['Vx']/1000, color='k', linewidth=0.8, label=r'V$_x$ (MAGE)')  # divide by 1000 to get in km/s
axes[5].set_ylabel(r'V$_x$ [km/s]')

# Plotting Bx, By, Bz IMF
axes[6].plot(data['time'], data['By'], label=r'B$_y^\text{IMF}$', color='k', linewidth=0.5)
axes[6].plot(data['time'], data['Bz'], label=r'B$_z^\text{IMF}$', color='m', linewidth=0.5)
axes[6].set_ylabel('[nT]')
axes[6].legend(loc='upper right', ncol=2)

plt_adjust(limits)
fig.align_ylabels(axes)

if not Both:
 savefig('_imf', 'imf_mage', logger)
 savefig_paper('imf_mage', logger)

if Both:
  # Reading Dean's data
  logger.info(f'Reading {dean_fname}')
  columns = ['year', 'month', 'day', 'hour', 'min', 'sec', 'msec', 'Bx[nT]', 'By[nT]', 'Bz[nT]', 'Vx[km/s]', 'Vy[km/s]', 'Vz[km/s]', 'N[cm^(-3)]', 'T[Kelvin]']
  data = pd.read_csv(dean_fname, delim_whitespace=True, names=columns, header=0)
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
  
  # Cropping data
  for key in df.columns:
    time, df[key] = subset(df['time'], df[key], limits['data'][0], limits['data'][1])
  df['time'] = time

  ## To plot Magnetosonic Mach, see:
  ##  https://omniweb.gsfc.nasa.gov/ftpbrowser/bow_derivation.html
  ##  https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2009JA014998

  # Adding T to existing axes
  axes[3].plot(df['time'], df['T']/1e6, label='T (Dean)', color='gray', linewidth=0.8)
  axes[3].legend(loc='upper right', ncol=2)

  # Adding Vx to existing axes
  axes[5].plot(df['time'], df['Vx'], label=r'V$_x$ (Dean)', color='gray', linewidth=0.8)
  axes[5].legend(loc='upper right', ncol=2)

  # Adding By and Bz to the existing axes
  axes[6].plot(df['time'], df['By'], label=r'B$_y^\text{Dean}$', color='gray', linewidth=0.8)
  axes[6].plot(df['time'], df['Bz'], label=r'B$_z^\text{Dean}$', color='dimgray', linewidth=0.5, linestyle='--')
  axes[6].legend(loc='upper right', ncol=4)

  savefig('_imf', 'imf_all', logger)

utilrsw.rm_if_empty('log/plot_imf.errors.log')