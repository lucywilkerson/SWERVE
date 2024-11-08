import os
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt

data_dir = os.path.join('..', '..', '2024-AGU-data', 'gic')
data_file = os.path.join(data_dir, 'gic_all.pkl')

def plot_save(fname):
  print(f"Saving {fname}.{png, pdf, svg}")
  plt.savefig(f'{fname}.png',dpi=300)
  plt.savefig(f'{fname}.pdf')
  plt.savefig(f'{fname}.svg')

def plot_combined(gic_all):

  dto = datetime.datetime(2024, 5, 10, 12, 0)
  dtf = datetime.datetime(2024, 5, 12, 0, 0)
  linewidth = 0.5

  out_dir = os.path.join(data_dir, 'plots')
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  # setting up 4-panel plot
    
  fig, axs = plt.subplots(2, 2,figsize=(23,10))
  #fig.suptitle('TVA calculated GIC',fontsize=20)

  locs = list(gic_all.keys())
  locs.remove('anderson')

  # Loop to create four subpanels with GIC model vs anderson
  for idx, loc in enumerate(locs):
    axs[idx].plot(gic_all[loc]['time'], gic_all[loc]['data'], linewidth=linewidth, label='Calculated')
    axs[idx].plot(gic_all['anderson']['time'], gic_all['anderson']['data'],linewidth=linewidth, label='Measured')
    axs[idx].set_title(f"TVA {loc}", fontsize=15) 
    axs[idx].set(xlabel='time', ylabel='GIC [A]')
    axs[idx].set_xlim(dto, dtf)
    axs[idx].set_ylim(-25, 45)
    axs[idx].legend()
    axs[idx].grid('--')
    axs[idx].label_outer()

  # code to save plots out_dir here
  #fname = use key from info, e.g., bullrun
  #fname_base = os.path.join(out_dir, fname)
  #plot_save(fname)

with open(data_file, 'rb') as f:
  gic_all = pickle.load(f)


plot_combined(gic_all)
