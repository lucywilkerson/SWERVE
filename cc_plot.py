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
plt.close()

avg_std = []
for idx, row in sorted_rows.iterrows():
  avg_std.append(np.mean([row['std_1'], row['std_2']]))

plt.scatter(avg_std, np.abs(sorted_rows['cc']))
plt.xlabel('Average standard deviation')
plt.ylabel('|cc|')
plt.grid(True)
#plt.show()

print('Saving cc_plot_std.svg')
plt.savefig('cc_plot_std.svg', bbox_inches='tight')
