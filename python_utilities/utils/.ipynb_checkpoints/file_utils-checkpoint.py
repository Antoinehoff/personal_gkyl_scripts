import os
import re
import numpy as np

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


def find_available_frames(sim_info,dataname='field'):
    # Regular expression pattern to match files with the format "*dataname_X.gkyl"
    pattern = re.compile(r"%s_([0-9]+)\.gkyl$"%dataname)
    folder_path = sim_info.simdir+sim_info.wkdir
    # List to store the frame numbers
    frames = []
    if len(os.listdir(folder_path)) == 0:
        print("No file found in %s"%folderpath)
    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        # Use regular expression to find matching filenames
        match = pattern.search(filename)
        if match:
            # Extract the frame number and add it to the list
            frame_number = int(match.group(1))
            frames.append(frame_number)

    # Sort the frame numbers for easier interpretation
    frames.sort()
    frames = list(set(frames))
    return frames
