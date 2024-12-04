#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:02:05 2024

@author: Dean Thomas
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as dates

data_dir      = os.path.join('..', '..', '2024-AGU-data')
data_dir_swmf = os.path.join( data_dir, 'SWMF')
data_dir_tva  = os.path.join(data_dir, 'tva', 'mag')

# info block with names of files for analysis
info = {
        "tva":  [ 'bullrun_mag_20240509.pkl', 'union_mag_20240509.pkl'],
        "gap":  [ 'dB_bs_gap-BullRun.pkl', 'dB_bs_gap-Union.pkl'],
        'iono': [ 'dB_bs_iono-BullRun.pkl', 'dB_bs_iono-Union.pkl'],
        'msph': [ 'dB_bs_msph-BullRun.pkl', 'dB_bs_msph-Union.pkl'],
        'title':[ 'Bull Run', 'Union']
      }

def readTVA( i, info, data_dir=data_dir_tva ):
    """ Read the TVA data
    
    inputs:
        i = which data in info block lists to use.
        
        info = info block defined above with the file names used in calculations
        
        data_dir = data directory
    
    outputs:
        dftva = dataframe with TVA data    
    """
    
    filetva = os.path.join(data_dir, info['tva' ][i])
    dftva = pd.read_pickle( filetva )
    
    # Pull data from TVA files. Note, we have total magnetic field from TVA,
    # but SWMF has only changes (delta B).  So we substract the mean B field
    # from the TVA data
    Bxmean = dftva['Bx'].mean()
    Bymean = dftva['By'].mean()
    Bzmean = dftva['Bz'].mean()
    
    dftva['dBx'] = dftva['Bx'] - Bxmean
    dftva['dBy'] = dftva['By'] - Bymean
    dftva['dBz'] = dftva['Bz'] - Bzmean
    
    # Since we're not sure of the magnetometer coordinates, we use B horizontal
    # Determine magnitude of B horizontal
    dftva['dBh'] = np.sqrt( (dftva['Bx'] - Bxmean)**2 + (dftva['By'] - Bymean)**2 )  
    
    return dftva
    
def readSWMF( i, info, data_dir=data_dir_swmf ):
    """ Read the SWMF data
    
    inputs:
        i = which data in info block lists to use.
        
        info = info block defined above with the file names used in calculations
        
        data_dir = data directory
    
    outputs:
        dftot = dataframe with SWMF data    
    """
     
    filegap  = os.path.join(data_dir, info['gap' ][i])
    fileiono = os.path.join(data_dir, info['iono'][i])
    filemsph = os.path.join(data_dir, info['msph'][i])
    
    dfgap  = pd.read_pickle( filegap )
    dfiono = pd.read_pickle( fileiono )
    dfmsph = pd.read_pickle( filemsph )
    
    dftot = pd.DataFrame()
    
    # Pull data from SWMF files. Note, total delta B is sum of gap, ionosphere Hall,
    # ionosphere Pedersen, and magnetospheric contributions
    dftot['Time'] = dfgap['Datetime']
    dftot['Bn'] = dfgap['Bn'] + dfiono['Bnh'] + dfiono['Bnp'] + dfmsph['Bn']
    dftot['Be'] = dfgap['Be'] + dfiono['Beh'] + dfiono['Bnp'] + dfmsph['Be']
    dftot['Bd'] = dfgap['Bd'] + dfiono['Bdh'] + dfiono['Bnp'] + dfmsph['Bd']
    
    # Determine magnitude of delta B horizontal
    dftot['dBh'] = np.sqrt( dftot['Bn']**2 + dftot['Be']**2 )
    
    return dftot
   
if __name__ == "__main__":
    
    # Loop through entries in info block
    # Generate plots comparing delta B_H from TVA magnetometers and SWMF simulation
    for i in range(len(info["tva"])):
        dftva = readTVA( i, info )
        dfswmf = readSWMF( i, info )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot( dftva['Time'], dftva['dBh'] )
        ax.plot( dfswmf['Time'], dfswmf['dBh'] )
        ax.set_ylabel(r'$|\delta B_H|$ (nT)')
        ax.set_xlabel('Time')
        ax.set_title( info['title'][i] )    
        date_form = dates.DateFormatter("%m-%d\n%H:%M")
        ax.xaxis.set_major_formatter(date_form)
        

        