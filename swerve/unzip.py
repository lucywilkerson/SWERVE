import zipfile
import os
from swerve import config

CONFIG = config()
data_dir = CONFIG['dirs']['data']
event = CONFIG['event']

def unzip_all_zip_files(directory_path, output_directory=None):
    """
    Unzips all .zip files found in a given directory.

    Args:
        directory_path (str): The path to the directory containing the zip files.
        output_directory (str, optional): The directory where the contents
                                          of the zip files will be extracted.
                                          If None, contents are extracted to
                                          a new folder with the zip file's name
                                          in the same directory as the zip file.
    """
    for item in os.listdir(directory_path):
        if item.endswith(".zip"):
            zip_file_path = os.path.join(directory_path, item)
            
            if output_directory:
                extraction_path = output_directory
            else:
                # Extract to a new folder named after the zip file (without .zip extension)
                zip_name_without_extension = os.path.splitext(item)[0]
                extraction_path = os.path.join(directory_path, zip_name_without_extension)
                os.makedirs(extraction_path, exist_ok=True) # Create the directory if it doesn't exist

            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
                print(f"Successfully unzipped '{item}' to '{extraction_path}'")
                os.remove(zip_file_path)
                print(f"Deleted original zip file: '{item}'.")
            except zipfile.BadZipFile:
                print(f"Error: '{item}' is a bad zip file and cannot be unzipped.")
            except Exception as e:
                print(f"An error occurred while unzipping '{item}': {e}")

for nerc_data_type in ['gic', 'mag']:
    gic_zip_dir = os.path.join(data_dir, 'data_original', 'nerc', event, nerc_data_type)
    unzip_all_zip_files(gic_zip_dir, gic_zip_dir)