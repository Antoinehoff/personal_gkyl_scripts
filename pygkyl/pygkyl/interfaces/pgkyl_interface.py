"""
pgkyl_interface.py

This file contains functions that are used to interface with the postgkyl library.

Functions:
- get_values: Get values from GData and handle dimensionality.
- interpolate: Interpolate data and handle dimensionality.
- get_grid: Get the grid from GData and handle dimensionality.
- get_cells: Get the cells from GData and handle dimensionality.
- integrate: Integrate the data using GData.

"""

import postgkyl as pg
import numpy as np
from ..utils import file_utils

def get_dg_and_gdata(filename:str, polyorder=1, polytype='ms'):
    if polytype == 'gkhyb':
        # retrieve the file prefix by removing from the end to the last hifen
        prefix = filename[:filename.rfind('-')]
        # find the species name, which is the three letters after the last hifen
        species = file_utils.find_species(filename)
        mapc2p_vel_name = prefix + '-' + species + '_mapc2p_vel.gkyl'
        jacobvel_name = prefix + '-' + species + '_jacobvel.gkyl'
        Gdata = pg.data.GData(filename, mapc2p_vel_name=mapc2p_vel_name)
        Jv_Gdata = pg.data.GData(jacobvel_name, mapc2p_vel_name=mapc2p_vel_name)
        Gdata._values = Gdata.get_values() / Jv_Gdata.get_values()
    else:
        Gdata = pg.data.GData(filename)
    dg = pg.data.GInterpModal(Gdata, poly_order=polyorder, basis_type=polytype)
    return dg, Gdata

def get_values(Gdata):
    # Extend dimension for 2D and 4D data. Pygkyl considers everythin as 3D or 5D
    if Gdata.get_values().ndim in [3, 5]:
        values = np.expand_dims(Gdata.get_values(), axis=1)
        return np.concatenate((values, values), axis=1)
    else:
        return Gdata.get_values()

def interpolate(Gdata,comp,polyorder=1, polytype='ms'):
    dg = pg.data.GInterpModal(Gdata, poly_order=polyorder, basis_type=polytype, periodic=False, num_interp=polyorder+1)
    values = dg.interpolate(comp)
    if values.ndim == 3:
        return values
    if values.ndim == 2:
        values = np.expand_dims(values, axis=1)
        return np.concatenate((values, values), axis=1)
    
def get_grid(Gdata):
    values = Gdata.get_grid()        
    if len(values) == 5 :
        return values
    elif len(values) == 4 :
        return [values[0], np.array([0, 1/3, 2/3]), values[1], values[2], values[3]]
    elif len(values) == 3 :
        return values
    elif len(values) == 2 :
        return [values[0], np.array([0, 1/3, 2/3]), values[1]]
    
def get_cells(Gdata):
    cells = Gdata.ctx['cells']
    if len(cells) in [3,5]:
        return cells
    elif len(cells) == 4:
        return [cells[0], 2, cells[1], cells[2], cells[3]]
    elif len(cells) == 2:
        cells = [cells[0], 2, cells[1]]
        return cells

def integrate(Gdata):
    return Gdata.integrate()

def get_gkyl_data(file):
    return pg.data.GData(file)

def get_gkyl_values(file,comp=0,polyorder=1,polytype='ms'):
    if comp is None:
        return get_values(pg.data.GData(file))
    else:
        return interpolate(pg.data.GData(file),comp=comp,polyorder=polyorder, polytype=polytype)

def get_gkyl_grid(file):
    return get_grid(pg.data.GData(file))

def get_gkyl_cells(file):
    return get_cells(pg.data.GData(file))

def file_exists(file):
    try:
        with open(file, 'r') as f:
            return True
    except FileNotFoundError:
        return False