import os
import pickle
from matplotlib import pyplot as plt

data_dir = os.path.join('..', 'data', 'SWMF')

data = {'gap': {
          'file': 'dB_bs_gap-FRD.pkl'
        },
        'iono': {
          'file': 'dB_bs_iono-FRD.pkl'
        },
        'msph': {
          'file': 'dB_bs_msph-FRD.pkl'
        }
      }

for key, region in data.items():
  file = os.path.join(data_dir, region['file'])
  print(f"Reading {file}")
  with open(file, 'rb') as f:
    data[key]['file_data'] = pickle.load(f)
    if key != 'iono':
      Bn = data[key]['file_data']['Bn']
    else:
      Bn = data[key]['file_data']['Bnp'] + data[key]['file_data']['Bnh']
    plt.plot(data[key]['file_data']['Datetime'], Bn, label=key)
    plt.legend()
    plt.show()