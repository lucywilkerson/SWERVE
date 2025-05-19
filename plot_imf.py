import os
import numpy as np
import h5py
from datetime import datetime, timedelta

from datetick import datetick

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600

data_dir = os.path.join('..', '2024-May-Storm-data')
base_dir = os.path.join(data_dir, '_processed')

def subset(time, data, start, stop):
  idx = np.logical_and(time >= start, time <= stop)
  if data.ndim == 1:
    return time[idx], data[idx]
  return time[idx], data[idx,:]

start = datetime(2024, 5, 10, 15, 0)
stop = datetime(2024, 5, 12, 6, 0)

def savefig(sid, fname, sub_dir="", fmts=['png','pdf']):
  fdir = os.path.join(base_dir, sid.lower().replace(' ', ''), sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    print(f"    Saving {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')

def savefig_paper(fname, sub_dir="", fmts=['png','pdf']):
  fdir = os.path.join('..','2024-May-Storm-paper', sub_dir)
  if not os.path.exists(fdir):
    os.makedirs(fdir)
  fname = os.path.join(fdir, fname)

  for fmt in fmts:
    print(f"    Saving {fname}.{fmt}")
    plt.savefig(f'{fname}.{fmt}', bbox_inches='tight')


# Open file
file_path = os.path.join(data_dir, 'mage', 'bcwind.h5')
print(f'Reading {file_path}')
imf_data = h5py.File(file_path,'r')
"""data_keys = imf_data.keys()
print(data_keys)
exit()"""

# Assigning variables
al = imf_data['al'][()]
ae = imf_data['ae'][()]
Kp = imf_data['Kp'][()]
symh = imf_data['symh'][()]
temp = imf_data['Temp'][()]
mach = imf_data['Magnetosonic Mach'][()]
Vx = imf_data['Vx'][()]
Bx = imf_data['Bx'][()]
By = imf_data['By'][()]
Bz = imf_data['Bz'][()]

# Convert time from MJD
mjd = imf_data['MJD'][()]
time_original = np.array([datetime(1858, 11, 17) + timedelta(days=day) for day in mjd])
time = time_original

#fig, axes = plt.subplots(7, 1, figsize=(8.5, 11))
plt.figure(figsize=(8.5, 11))
gs = plt.gcf().add_gridspec(7, 1)
axes = gs.subplots(sharex=True)

# Plotting al and ae

for i in range(al.size):
  if time_original[i] <= datetime(2024, 5, 10, 0, 0):
    al[i] = np.nan
    ae[i] = np.nan

time, al = subset(time_original, al, start, stop)
axes[0].plot(time,-al, label=r'$-$AL', color='k', linewidth=1)
time, ae = subset(time_original, ae, start, stop)
axes[0].plot(time,ae, label='AE', color='m', linewidth=0.5)
axes[0].set_ylabel('[nT]')
axes[0].legend(ncol=2)

# Plotting Kp
kp_times = []
kp_values = []
# Creating a 1-hour grid from start to stop
current_time = start
while current_time <= stop:
  # Finding the index of the closest Kp value (Kp is every 3 hours, centered at :30)
  # Find the previous 3-hour interval
  kp_idx = None
  for i, t in enumerate(time_original):
    if t.hour % 3 == 0 and t.minute == 0:
      kp_time_center = t + timedelta(hours=1.5)
      # If current_time falls within this 3-hour interval, use this Kp
      if kp_time_center - timedelta(hours=1.5) <= current_time < kp_time_center + timedelta(hours=1.5):
        kp_idx = i
        break
  if kp_idx is not None:
    kp_times.append(current_time)
    kp_values.append(Kp[kp_idx])
  current_time += timedelta(hours=1)

#kp_times = np.array(kp_times)
#kp_values = np.array(kp_values)
#kp_times, kp_values = subset(kp_times, kp_values, start, stop)
axes[1].step(kp_times, kp_values, where='post', color='k')
axes[1].fill_between(kp_times, kp_values, 0, step='post', color='k')
axes[1].set_ylabel(r'K$_p$')
axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[1].set_ylim(4.5, 9.5)

# Plotting symh
time, symh = subset(time_original, symh, start, stop)
axes[2].plot(time, symh, color='k', linewidth=0.8)
axes[2].set_ylabel('SYM-H [nT]')

# Plotting temperature
time, temp = subset(time_original, temp, start, stop)
axes[3].plot(time, temp / 1e6, color='k', linewidth=0.8)
axes[3].set_ylabel(r'T [MK]')

# Plotting mach
time, mach = subset(time_original, mach, start, stop)
axes[4].plot(time, mach, color='k', linewidth=0.8)
axes[4].set_ylabel(r'Mach')

# Plotting Vx
time, Vx = subset(time_original, Vx, start, stop)
axes[5].plot(time, Vx/1000, color='k', linewidth=0.8)  # divide by 1000 to get in km/s
axes[5].set_ylabel(r'V$_x$ [km/s]')

# Plotting Bx, By, Bz IMF
#time, Bx = subset(time_original, Bx, start, stop)
#axes[6].plot(time, Bx, label=r'B$_x^\text{IMF}$', linewidth=0.5)
time, By = subset(time_original, By, start, stop)
axes[6].plot(time, By, label=r'B$_y^\text{IMF}$', color='k', linewidth=0.5)
time, Bz = subset(time_original, Bz, start, stop)
axes[6].plot(time, Bz, label=r'B$_z^\text{IMF}$', color='m', linewidth=0.5)
axes[6].set_ylabel('[nT]')
axes[6].legend(loc='upper right', ncol=2)

xlims = [datetime(2024, 5, 10, 11, 0), stop]
#plt.gca().set_xlim(datetime.datetime(2024, 5, 10, 11, 0), stop)
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

savefig('_imf','imf_mage')
savefig_paper('imf_mage')
#plt.show()