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
#data_keys = imf_data.keys()
#print(data_keys)

# Assigning variables
al = imf_data['al'][()]
ae = imf_data['ae'][()]
Kp = imf_data['Kp'][()]
symh = imf_data['symh'][()]
Vx = imf_data['Vx'][()]
Bx = imf_data['Bx'][()]
By = imf_data['By'][()]
Bz = imf_data['Bz'][()]

# Convert time from MJD
mjd = imf_data['MJD'][()]
time_original = np.array([datetime(1858, 11, 17) + timedelta(days=day) for day in mjd])
time = time_original

fig, axes = plt.subplots(5, 1, figsize=(8.5, 11))

# Plotting al and ae

for i in range(al.size):
  if time_original[i] <= datetime(2024, 5, 10, 0, 0):
    al[i] = np.nan
    ae[i] = np.nan

#time, al = subset(time_original, al, start, stop)
axes[0].plot(time,al, label='AL')
#time, ae = subset(time_original, ae, start, stop)
axes[0].plot(time,ae, label='AE')
axes[0].set_ylabel('[nT]')
axes[0].legend()

# Plotting Kp
kp_times = []
kp_values = []
# Extract Kp values at 3-hour increments (midnight, 3am, 6am, etc.)
for i, t in enumerate(time_original):
  if t.hour % 3 == 0 and t.minute == 0:
    kp_times.append(t + timedelta(hours=1.5))  # Center bars at 1:30am, 4:30am, etc.
    kp_values.append(Kp[i])

#kp_times = np.array(kp_times)
#kp_values = np.array(kp_values)
#kp_times, kp_values = subset(kp_times, kp_values, start, stop)
axes[1].step(kp_times, kp_values, where='mid')
axes[1].set_ylabel(r'K$_p$')
axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[1].set_ylim(5.5, 9.5)

# Plotting symh
#time, symh = subset(time_original, symh, start, stop)
axes[2].plot(time,symh)
axes[2].set_ylabel('SYM-H [nT]')

# Plotting Vx
#time, Vx = subset(time_original, Vx, start, stop)
axes[3].plot(time,Vx/1000) # divide by 1000 to get in km/s
axes[3].set_ylabel(r'V$_x$ [km/s]')

# Plotting Bx, By, Bz IMF
#time, Bx = subset(time_original, Bx, start, stop)
#axes[4].plot(time, Bx, label=r'B$_x^\text{IMF}$', linewidth=0.5)
#time, By = subset(time_original, By, start, stop)
axes[4].plot(time, By, label=r'B$_y^\text{IMF}$', linewidth=0.5)
#time, Bz = subset(time_original, Bz, start, stop)
axes[4].plot(time, Bz, label=r'B$_z^\text{IMF}$', linewidth=0.5)
axes[4].set_ylabel('[nT]')
axes[4].legend(loc='upper right')

xlims = [start, stop]
for ax in axes:
  ax.set_xlim(xlims)
  ax.grid(True)
  datetick('x', axes=ax)

# Remove x-axis labels for all subplots except the bottom one
for ax in axes[:-1]:
  ax.set_xticklabels([])

savefig('_imf','imf_mage')
savefig_paper('imf_mage')
#plt.show()