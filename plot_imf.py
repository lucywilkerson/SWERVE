import os
import h5py
from datetime import datetime, timedelta

from datetick import datetick

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600

data_dir = os.path.join('..', '2024-May-Storm-data')
file_path = os.path.join(data_dir, 'mage', 'bcwind.h5')

# Open file
imf_data = h5py.File(file_path,'r')
data_keys = imf_data.keys()
print(data_keys)

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

fig, axes = plt.subplots(5,1)

# Plotting al and ae
axes[0].plot(time,al)
axes[0].plot(time,ae)

# Plotting Kp
axes[1].plot(time,Kp)

# Plotting symh
axes[2].plot(time,symh)

# Plotting Vx
axes[3].plot(time,Vx)

# Plotting Bx, By, Bz IMF
axes[4].plot(time,Bx)
axes[4].plot(time,By)
axes[4].plot(time,Bz)

# Remove x-axis labels for all subplots except the bottom one
for ax in axes[:-1]:
    ax.set_xticklabels([])

# Add grid to all axes
for ax in axes:
    ax.grid(True)

datetick()
plt.show()