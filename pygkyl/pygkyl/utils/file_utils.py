"""
file_utils.py

This module provides various utilities for file handling.

Functions:
- find_prefix: Extracts file prefixes from lua files.
- find_available_frames: Finds available frames in the simulation data directory.
- check_latex_installed: Checks if LaTeX is installed on the system.

"""

import fnmatch
import os,re,sys
from ..interfaces.flaninterface import FlanInterface
import postgkyl as pg

# function to extract filePrefix from lua file(s)
def find_prefix(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                #result.append(os.path.join(root, name))
                prefix = re.sub('.lua','',name)
                result.append(prefix)
    return result

def find_species(filename):
    # retrieve the file prefix by removing from the end to the last hifen
    prefix = filename[:filename.rfind('-')]
    # find the species name, which is the three letters after the last hifen
    species = filename[filename.rfind('-')+1:filename.rfind('-')+4]
    return species

def find_available_frames(simulation,dataname='field'):
    if simulation.code == 'gyacomo':
        frames = simulation.gyac.get_available_frames(dataname)
    elif dataname == 'flan':
        flan = FlanInterface(simulation.flandatapath)
        frames = flan.avail_frames
    else:
        # Regular expression pattern to match files with the format "prefix-dataname_framenumber.gkyl"
        pattern = re.compile(r"%s-%s_([0-9]+)\.gkyl$"%(simulation.prefix, dataname))
        folder_path = simulation.datadir
        # List to store the frame numbers
        frames = []
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print("Directory not found: %s"%folder_path)
            return frames
        
        if not os.path.isdir(folder_path):
            print("Path is not a directory: %s"%folder_path)
            return frames
        
        try:
            files = os.listdir(folder_path)
        except (PermissionError, OSError) as e:
            print("Error accessing directory %s: %s"%(folder_path, str(e)))
            return frames
        
        if len(files) == 0:
            print("No file found in %s"%folder_path)
            return frames
            
        # Iterate over all files in the specified folder
        for filename in files:
            # Skip directories, only process files
            if os.path.isdir(os.path.join(folder_path, filename)):
                continue
            # Use regular expression to find matching filenames
            match = pattern.search(filename)
            if match:
                # Extract the frame number and add it to the list
                frame_number = int(match.group(1))
                frames.append(frame_number)
                
        # if len(frames) == 0:
        #     print("No frames found for dataname '%s' in directory %s"%(dataname, folder_path))
        #     print("(Pattern used: %s-%s_([0-9]+).gkyl)"%(simulation.prefix, dataname))
                
    # Sort the frame numbers for easier interpretation
    frames = list(set(frames))
    frames.sort()
    
    return frames

import subprocess

def check_latex_installed(verbose=False):
    try:
        # Try running the 'latex --version' command
        result = subprocess.run(['latex', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if the command ran successfully
        if result.returncode == 0:
            if verbose:
                print("LaTeX is installed.")
                print(result.stdout.decode('utf-8'))  # Optional: Print the LaTeX version
            return True
        else:
            if verbose:
                print("LaTeX is not installed.")
                print(result.stderr.decode('utf-8'))  # Optional: Print the error message
            return False
    except FileNotFoundError:
        if verbose:
            print("LaTeX is not installed or not found in your system's PATH.")
        return False
    
def does_file_exist(fileIn):
  return os.path.exists(fileIn)