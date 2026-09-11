from swerve import config, sids, sids_and_events

import shutil
import os

import sys

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
data_dir = CONFIG['dirs']['data']

sids_only = None # Read all sites.

if len(CONFIG['event']) > 1:
    sids_only = sids_and_events(key=sids_only, data_type='GIC', data_class='measured')
else:
    sids_only = sids(key=sids_only, data_type='GIC', data_class='measured', add_event=True)

for sid, event in sids_only:
    event_dir = os.path.join(data_dir, 'data_processed', event)
    sid = sid.lower().replace(' ', '')
    # Define source and destination paths
    source_image = os.path.join(event_dir,'sites',sid,'figures','original','GIC_measured_NERC.png')
    if not os.path.isfile(source_image):
        #source_image = os.path.join(event_dir,'sites',sid,'figures','original','GIC_measured_TVA.png')
        continue
    destination_folder = os.path.join(event_dir,'_all','all_gic')
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




