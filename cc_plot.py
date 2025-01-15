import os
import pickle
import numpy as np

data_dir = os.path.join('..', '2024-AGU-data')
pkl_file = os.path.join(data_dir, '_results', 'cc.pkl')
with open(pkl_file, 'rb') as file:
  sorted_rows = pickle.load(file)

import matplotlib.pyplot as plt

plt.scatter(sorted_rows['dist(km)'], np.abs(sorted_rows['cc']))
plt.xlabel('Distance (km)')
plt.ylabel('|cc|')
plt.grid(True)
#plt.show()

print('Saving cc_plot.svg')
plt.savefig('cc_plot.svg', bbox_inches='tight')
