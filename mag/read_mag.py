#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:02:24 2024

@author: Dean Thomas
"""

import os
import csv
import numpy
import pandas
import datetime

# .csv files are 1-sec cadence for magnetic field
data_dir = os.path.join('..', '..', '2024-AGU-data')
data_dir_mag = os.path.join(data_dir, 'tva', 'mag', 'orig' )
data_dir_save = os.path.join(data_dir, 'tva', 'mag' )

# info block defines TVA files to be processed
info = {
        "files": ["bullrun_mag_20240509.csv", "union_mag_20240509.csv"]
} 

def read(file,data_dir=data_dir_mag):
    """ Read the TVA data
    
    inputs:
        file = file to be parsed
        
        data_dir = data directory
    
    outputs:
        bx,by,bz,b,time = numpy arrays with magnetic field and time data   
    """

    if file.endswith('.csv'):
        bx = []
        by = []
        bz = []
        b  = []
        time = []
      
        filepath = os.path.join(data_dir, file)
        print(f"Reading {file}")
        with open(filepath,'r') as csvfile:
          rows = csv.reader(csvfile, delimiter = ',')
          next(rows)  # Skip header row.
          for row in rows:
              time.append(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
              bx.append(float(row[1]))
              by.append(float(row[2]))
              bz.append(float(row[3]))
              bmag = numpy.sqrt( float(row[1])**2 + float(row[2])**2 + float(row[3])**2 )
              b.append(float(bmag))
  
    return numpy.array(bx), numpy.array(by), numpy.array(bz), numpy.array(b), time

def save( file, bx, by, bz, b, time, data_dir=data_dir_save ):
    """ Save the parsed TVA data
    
    inputs:
        file = file that was parsed, used to generate pickle file name
        
        data_dir = data directory where pickle file will be saved
    
    outputs:
        None except saved pickle file   
    """
    
    filepath =  os.path.splitext(file)[0] + '.pkl'
    print(f"Saving {filepath}")
    filepath = os.path.join(data_dir, filepath)
    
    df = pandas.DataFrame()
    df['Time'] = time
    df['Bx'] = bx
    df['By'] = by
    df['Bz'] = bz
    df['B']  = b
    
    df.to_pickle( filepath )
    return 0
    
if __name__ == "__main__":
    
    # Loop through files in info, defined above
    # Parse TVA data and save to pickle files
    for file in info['files']:
        bx, by, bz, b, time = read(file)
        save( file, bx, by, bx, b, time )