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
        # Regular expression pattern to match files with the format "*dataname_X.gkyl"
        pattern = re.compile(r"%s_([0-9]+)\.gkyl$"%dataname)
        folder_path = simulation.data_param.datadir
        # List to store the frame numbers
        frames = []
        if len(os.listdir(folder_path)) == 0:
            print("No file found in %s"%folder_path)
        # Iterate over all files in the specified folder
        for filename in os.listdir(folder_path):
            # Use regular expression to find matching filenames
            match = pattern.search(filename)
            if match:
                # Extract the frame number and add it to the list
                frame_number = int(match.group(1))
                frames.append(frame_number)
                
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