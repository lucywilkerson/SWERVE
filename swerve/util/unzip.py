import os
import zipfile


def unzip():
    """Unzip all configured NERC GIC and magnetometer archives."""
    from swerve import config

    def _unzip_all_zip_files(directory_path, output_directory=None):
        """Unzip and remove all zip files found in a directory."""
        for item in os.listdir(directory_path):
            if not item.endswith(".zip"):
                continue
            zip_file_path = os.path.join(directory_path, item)
            if output_directory:
                extraction_path = output_directory
            else:
                zip_name_without_extension = os.path.splitext(item)[0]
                extraction_path = os.path.join(directory_path, zip_name_without_extension)
                os.makedirs(extraction_path, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
                print(f"Successfully unzipped '{item}' to '{extraction_path}'")
                os.remove(zip_file_path)
                print(f"Deleted original zip file: '{item}'.")
            except zipfile.BadZipFile:
                print(f"Error: '{item}' is a bad zip file and cannot be unzipped.")
            except Exception as error:
                print(f"An error occurred while unzipping '{item}': {error}")

    CONFIG = config()
    data_dir = CONFIG['dirs']['data']
    events = CONFIG['event']
    for nerc_data_type in ['gic', 'mag']:
        for event in events:
            print(f'Unzipping NERC files for event {event}')
            zip_dir = os.path.join(data_dir, 'data_original', 'nerc', event, nerc_data_type)
            _unzip_all_zip_files(zip_dir, zip_dir)


if __name__ == '__main__':
    unzip()