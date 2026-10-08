import os
import shutil


def move_pics():
    """Collect configured NERC measured-GIC plots into each event summary directory."""
    from swerve import config, sids, sids_and_events

    CONFIG = config()
    data_dir = CONFIG['dirs']['data']
    sids_only = None
    if len(CONFIG['event']) > 1:
        sids_only = sids_and_events(key=sids_only, data_type='GIC', data_class='measured')
    else:
        sids_only = sids(key=sids_only, data_type='GIC', data_class='measured', add_event=True)

    for sid, event in sids_only:
        event_dir = os.path.join(data_dir, 'data_processed', event)
        sid = sid.lower().replace(' ', '')
        source_image = os.path.join(event_dir, 'sites', sid, 'figures', 'original', 'GIC_measured_NERC.png')
        if not os.path.isfile(source_image):
            continue
        destination_folder = os.path.join(event_dir, '_all', 'all_gic')
        new_fname = f'{sid}_GIC_measured.png'
        os.makedirs(destination_folder, exist_ok=True)

        try:
            shutil.copy(source_image, os.path.join(destination_folder, new_fname))
            print(f"'{source_image}' copied successfully to '{destination_folder}'")
        except FileNotFoundError:
            print(f"Error: Source file '{source_image}' not found.")
        except Exception as error:
            print(f"An error occurred: {error}")


if __name__ == '__main__':
    move_pics()