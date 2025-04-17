import os
import numpy as np
import h5py
from datetime import datetime, timedelta

from datetick import datetick

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600

data_dir = os.path.join('..', '2024-May-Storm-data')
base_dir = os.path.join(data_dir, '_processed')

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
time = [datetime(1858, 11, 17) + timedelta(days=day) for day in mjd]

fig, axes = plt.subplots(5,1,figsize=(8.5, 11))

# Plotting al and ae
for i in range(al.size):
  if time[i] <= datetime(2024, 5, 10, 0, 0):
    al[i] = np.nan
    ae[i] = np.nan

axes[0].plot(time,al, label='AL')
axes[0].plot(time,ae, label='AE')
axes[0].set_ylabel('[nT]')
axes[0].legend()

# Plotting Kp
kp_times = []
kp_values = []
# Extract Kp values at 3-hour increments (midnight, 3am, 6am, etc.)
for i, t in enumerate(time):
  if t.hour % 3 == 0 and t.minute == 0:
    kp_times.append(t + timedelta(hours=1.5))  # Center bars at 1:30am, 4:30am, etc.
    kp_values.append(Kp[i])

axes[1].step(kp_times, kp_values, where='mid')
axes[1].set_ylabel(r'K$_p$')

# Plotting symh
axes[2].plot(time,symh)
axes[2].set_ylabel('SYM-H [nT]')

# Plotting Vx
axes[3].plot(time,Vx/1000) # divide by 1000 to get in km/s
axes[3].set_ylabel(r'V$_x$ [km/s]')

# Plotting Bx, By, Bz IMF
#axes[4].plot(time, Bx, label=r'B$_x^\text{IMF}$', linewidth=0.5)
axes[4].plot(time, By, label=r'B$_y^\text{IMF}$', linewidth=0.5)
axes[4].plot(time, Bz, label=r'B$_z^\text{IMF}$', linewidth=0.5)
axes[4].set_ylabel('[nT]')
axes[4].legend(loc='lower left')

# Remove x-axis labels for all subplots except the bottom one
for ax in axes[:-1]:
    ax.set_xticklabels([])

xlims = [time[0], time[-1]]
for ax in axes:
  ax.set_xlim(xlims)
  ax.grid(True)
datetick()

savefig('_imf','imf_mage')
savefig_paper('imf_mage')
#plt.show()