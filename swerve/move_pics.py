from swerve import config, sids

import shutil
import os

import sys

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
data_dir = CONFIG['dirs']['data']

sids_only = None # Read all sites.

sids_only = sids(key=sids_only, data_type='GIC', data_class='measured')

for sid in sids_only:
    sid = sid.lower().replace(' ', '')
    # Define source and destination paths
    source_image = os.path.join(data_dir,'data_processed','sites',sid,'figures','original','GIC_measured_NERC.png')
    if not os.path.isfile(source_image):
        source_image = os.path.join(data_dir,'data_processed','sites',sid,'figures','original','GIC_measured_TVA.png')
    destination_folder = os.path.join(data_dir, '_all','all_gic')
    new_fname = f'{sid}_GIC_measured.png'

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Copy the image
    try:
        shutil.copy(source_image, os.path.join(destination_folder,new_fname))
        print(f"'{source_image}' copied successfully to '{destination_folder}'")
    except FileNotFoundError:
        print(f"Error: Source file '{source_image}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")




