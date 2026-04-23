import json 
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
 
def read_json_file(filepath):
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
    
    except:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())  # parse each line as a JSON object
                data.append(record)
            
        return data

def write_manifest(output_filepath, final_list):
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        for entry in final_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            
def read_txt_from_file(txt_filepath):
    ''' Read contents of a txt file '''
    with open(txt_filepath, "r") as f:
        
        return f.read()
    
def get_all_files_in_directory(directory):
    ''' Get all files in a directory '''

    return [f for f in listdir(directory) if isfile(join(directory, f))]

def get_all_files_with_extension(directory, extension=".wav"):
    """Recursively get all files with a specific extension from a directory and its subdirectories."""
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension.lower()):
                matching_files.append(os.path.join(root, file))
    return matching_files

def delete_file(filepath):
    '''
    delete a file given a filepath
    '''
    try:
        if os.path.isfile(filepath):
            os.remove(filepath)
            print(f"Deleted: {filepath}")
        else:
            print(f"File not found or not a file: {filepath}")
    except Exception as e:
        print(f"Error deleting file {filepath}: {e}")
        
def export_rttm_file(list_rttm_entries, file_id, output_path = 'test.rttm'):
    ''' 
    Generate rttm file from list [[start, end, speaker]]
    '''
    for entry in list_rttm_entries:

        duration = entry[1] - entry[0]

        with open(output_path, 'a') as f:
            line = f"SPEAKER {file_id} 1 {entry[0]:.3f} {duration:.3f} <NA> <NA> {entry[2]} <NA>\n"
            f.write(line)
            
def get_folders(directory_path):
    """
    Gets all subdirectories within a given directory using os module functions.

    Args:
        directory_path (str): The path to the directory to search.

    Returns:
        list: A list of full paths to the subdirectories.
    """
    folders = []
    try:
        # List all entries in the directory
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    folders.append(entry.path)
    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")
    return folders

def ensure_folder_exists_os(folder_path):
    """
    Checks if a folder exists using the os module, and creates it if it doesn't.

    Args:
        folder_path (str): The path to the folder to check/create.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{folder_path}': {e}")
    else:
        print(f"Folder '{folder_path}' already exists.")
        
import os

def check_file_exists_os(file_path):
    """
    Checks if a file exists using the os module, and creates it if it doesn't.

    Args:
        file_path (str): The path to the file to check/create.
    """
    return os.path.exists(file_path)
        
def ensure_folders_to_file_exist(filepath):
    
    
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)
    
    return
    

def get_all_wav_files_in_folder(folderpath:str):
    
    folder = Path(folderpath)
    return [str(f) for f in folder.rglob("*.wav")]

