import os
import numpy as np
import h5py
from datetime import datetime, timedelta

from datetick import datetick

from storminator import DATA_DIR, LOG_DIR, plt_config, savefig, savefig_paper, subset

import utilrsw
logger = utilrsw.logger(log_dir=LOG_DIR)

import matplotlib.pyplot as plt


def read():
  file_path = os.path.join(DATA_DIR, 'mage', 'bcwind.h5')
  logger.info(f'Reading {file_path}')
  h5file = h5py.File(file_path,'r')

  data = {}
  for key in h5file.keys():
    logger.info(f'  Found {key}')
    data[key] = h5file[key][()]

  data['time'] = np.array([datetime(1858, 11, 17) + timedelta(days=day) for day in data['MJD']])

  for i in range(data['al'].size):
    if data['time'][i] <= datetime(2024, 5, 10, 0, 0):
      data['al'][i] = np.nan
      data['al'][i] = np.nan
  return data

data = read()

limits = plt_config()

plt.figure(figsize=(8.5, 11))
gs = plt.gcf().add_gridspec(7, 1)
axes = gs.subplots(sharex=True)

# Plotting al and ae
time, al = subset(data['time'], data['al'], limits['data'][0], limits['data'][1])
axes[0].plot(time,-al, label=r'$-$AL', color='k', linewidth=1)
time, ae = subset(data['time'], data['ae'], limits['data'][0], limits['data'][1])
axes[0].plot(time,ae, label='AE', color='m', linewidth=0.5)
axes[0].set_ylabel('[nT]')
axes[0].legend(ncol=2)

# Plotting Kp
# TODO: Should be able to do this without the loop (using only plt.stairs or plt.step).
kp_times = []
kp_values = []
# Creating a 1-hour grid from limits['data'][0] to limits['data'][1]
current_time = limits['data'][0]
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
axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[1].set_ylim(4.5, 9.5)

# Plotting symh
time, symh = subset(data['time'], data['symh'], limits['data'][0], limits['data'][1])
axes[2].plot(time, symh, color='k', linewidth=0.8)
axes[2].set_ylabel('SYM-H [nT]')

# Plotting temperature
time, temp = subset(data['time'], data['Temp'], limits['data'][0], limits['data'][1])
axes[3].plot(time, temp / 1e6, color='k', linewidth=0.8)
axes[3].set_ylabel(r'T [MK]')

# Plotting mach
time, mach = subset(data['time'], data['Magnetosonic Mach'], limits['data'][0], limits['data'][1])
axes[4].plot(time, mach, color='k', linewidth=0.8)
axes[4].set_ylabel(r'Mach')

# Plotting Vx
time, Vx = subset(data['time'], data['Vx'], limits['data'][0], limits['data'][1])
axes[5].plot(time, Vx/1000, color='k', linewidth=0.8)  # divide by 1000 to get in km/s
axes[5].set_ylabel(r'V$_x$ [km/s]')

# Plotting Bx, By, Bz IMF
time, By = subset(data['time'], data['By'], limits['data'][0], limits['data'][1])
axes[6].plot(time, By, label=r'B$_y^\text{IMF}$', color='k', linewidth=0.5)
time, Bz = subset(data['time'], data['Bz'], limits['data'][0], limits['data'][1])
axes[6].plot(time, Bz, label=r'B$_z^\text{IMF}$', color='m', linewidth=0.5)
axes[6].set_ylabel('[nT]')
axes[6].legend(loc='upper right', ncol=2)

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

savefig('_imf', 'imf_mage', logger)
savefig_paper('imf_mage', logger)
