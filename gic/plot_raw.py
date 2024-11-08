import os
import pickle
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
  

  # Loop to create four subpanels with GIC model vs anderson

  # Plot code here


  # code to save plots out_dir here
  # fname = use key from info, e.g., bullrun
  #fname_base = os.path.join(out_dir, fname)
  #plot_save(fname)

with open(data_file, 'rb') as f:
  gic_all = pickle.load(f)

plot_combined(gic_all)
