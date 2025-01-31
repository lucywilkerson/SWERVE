import os
import pickle
import numpy as np

import matplotlib.pyplot as plt

data_dir = os.path.join('..', '2024-AGU-data')
results_dir = os.path.join('..', '2024-AGU-data', '_results')

pkl_file = os.path.join(results_dir, 'cc.pkl')
print(f"Reading {pkl_file}")
with open(pkl_file, 'rb') as file:
  df = pickle.load(file)

def savefig(fname):
  if not os.path.exists(os.path.dirname(fname)):
    os.makedirs(os.path.dirname(fname))

  print(f"Writing {fname}.png")
  plt.savefig(f'{fname}.png', dpi=600, bbox_inches="tight")

plt.scatter(df['dist(km)'], np.abs(df['cc']))
plt.xlabel('Distance (km)')
plt.ylabel('|cc|')
plt.grid(True)
savefig(os.path.join(results_dir, 'cc_vs_dist_scatter'))
plt.close()

avg_std = np.mean(df[['std_1', 'std_2']], axis=1)
plt.scatter(avg_std, np.abs(df['cc']))
plt.xlabel('Average standard deviation (A)')
plt.ylabel('|cc|')
plt.grid(True)
savefig(os.path.join(results_dir, 'cc_vs_ave_std_scatter'))
plt.close()

# plot_maps
# plot_scatter
# plot_timeseries