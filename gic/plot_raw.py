import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

data_dir = os.path.join('..', '..', '2024-AGU-data', 'gic')
data_file = os.path.join(data_dir, 'gic_all.pkl')

def plot_save(fname):
  print(f"Saving {fname}.{png, pdf, svg}")
  plt.savefig(f'{fname}.png',dpi=300)
  plt.savefig(f'{fname}.pdf')
  plt.savefig(f'{fname}.svg')

def plot_combined(gic_all):
  out_dir = os.path.join(data_dir, 'plots')
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  # setting up 4-panel plot
    
  fig, axs = plt.subplots(2, 2,figsize=(23,10))
  fig.suptitle('TVA calculated GIC',fontsize=20)
  
  # Loop to create four subpanels with GIC model vs anderson
  for i in key:
    if i != 'anderson':
      for j,ax in enumerate(axs.flat):
          loc = key[j+1]
          if loc == i:
            ax.plot(gic_all[loc]['time'], gic_all[loc]['data'],linewidth=.5,label='Calculated data')
            ax.set_title(i,fontsize=15) 
    else:
      for ax in (axs.flat):
        ax.plot(gic_all[i]['time'], gic_all[i]['data'],linewidth=.5,label='Measured data')
         
  # Plot code here
  for ax in axs.flat:
    ax.set(xlabel='time', ylabel='GIC (Amps)')
    ax.set_xlim(datetime.datetime(2024, 5, 10, 12, 0),datetime.datetime(2024, 5, 12, 0, 0))
    ax.set_ylim(-25,45)
    ax.legend()
    ax.grid('--')
    ax.label_outer()

  # code to save plots out_dir here
  #fname = use key from info, e.g., bullrun
  #fname_base = os.path.join(out_dir, fname)
  #plot_save(fname)

with open(data_file, 'rb') as f:
  gic_all = pickle.load(f)

key = ['anderson', 'bullrun', 'montgomery', 'union', 'widowscreek']
item = ['data', 'files', 'time']

plot_combined(gic_all)
