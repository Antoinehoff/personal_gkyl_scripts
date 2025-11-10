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
from ..tools import gkhyb_basis as bf

def get_dg_and_gdata(filename:str, polyorder=1, polytype='ms'):
    if polytype == 'gkhyb':
        # retrieve the file prefix by removing from the end to the last hifen
        prefix = filename[:filename.rfind('-')]
        # find the species name, which is the three letters after the last hifen
        species = file_utils.find_species(filename)
        mapc2p_vel_name = prefix + '-' + species + '_mapc2p_vel.gkyl'
        Gdata = pg.data.GData(filename, mapc2p_vel_name=mapc2p_vel_name)
        Nbasis = Gdata.get_values().shape[-1]
        
        jacobvel_name = prefix + '-' + species + '_jacobvel.gkyl'
        Jv_Gdata = pg.data.GData(jacobvel_name, mapc2p_vel_name=mapc2p_vel_name)
        Gdata._values = Gdata.get_values() / Jv_Gdata.get_values() / np.sqrt(Nbasis)
    else:
        Gdata = pg.data.GData(filename)
    dg = pg.data.GInterpModal(Gdata, poly_order=polyorder, basis_type=polytype)
    return dg, Gdata

def get_values(Gdata):
    
    Gvalues = Gdata.get_values()
    # In the following we extend the dimensionality to always have
    # 5 dimensions in phase space and 3 dimensions in configuration space.
    if Gdata.ctx['basis_type'] == 'gkhybrid': # phase space data
        if Gvalues.ndim == 6: #3x2v + 1p data
            return Gvalues
        if Gvalues.ndim == 5: #2x2v + 1p data
            values = np.expand_dims(Gvalues, axis=1)
            return np.concatenate((values, values), axis=1)
        elif Gvalues.ndim == 4: #1x2v + 1p data
            values = np.expand_dims(Gvalues, axis=0)
            values = np.expand_dims(values, axis=0)
            values = np.concatenate((values, values), axis=0)
            return np.concatenate((values, values), axis=1)
    elif Gdata.ctx['basis_type'] == 'serendipity': # configuration space data
        if Gvalues.ndim == 4: #3x + 1p data
            return Gvalues
        if Gvalues.ndim == 3: #2x + 1p data
            values = np.expand_dims(Gvalues, axis=1)
            return np.concatenate((values, values), axis=1)
        elif Gvalues.ndim == 2: #1x + 1p data
            values = np.expand_dims(Gvalues, axis=0)
            values = np.expand_dims(values, axis=0)
            values = np.concatenate((values, values), axis=0)
            return np.concatenate((values, values), axis=1)
    else: # This is e.g. integrated data and time traces.
        return Gvalues

def interpolate(Gdata,comp,polyorder=1, polytype='ms'):
    dg = pg.data.GInterpModal(Gdata, poly_order=polyorder, basis_type=polytype, periodic=False, num_interp=polyorder+1)
    values = dg.interpolate(comp)
    if values.ndim == 3:
        return values
    if values.ndim == 2:
        values = np.expand_dims(values, axis=1)
        return np.concatenate((values, values), axis=1)

def get_interpolated_values(filename:str, comp=0, polyorder=1, polytype='ms'):
    dg, Gdata = get_dg_and_gdata(filename, polyorder=polyorder, polytype=polytype)
    return interpolate(Gdata, comp, polyorder, polytype)

def get_grid(Gdata):
    values = Gdata.get_grid()   
    if Gdata.ctx['basis_type'] == 'gkhybrid': # phase space data
        if len(values) == 5 :
            return values
        elif len(values) == 4 :
            return [values[0], np.array([0, 1/3, 2/3]), values[1], values[2], values[3]]
        elif len(values) == 3 :
            return [np.array([0, 1/3, 2/3]), np.array([0, 1/3, 2/3]), values[0], values[1], values[2]]
    elif Gdata.ctx['basis_type'] == 'serendipity': # configuration space data
        if len(values) == 3 :
            return values
        elif len(values) == 2 :
            return [values[0], np.array([0, 1/3, 2/3]), values[1]]
        elif len(values) == 1 :
            return [np.array([0, 1/3, 2/3]), np.array([0, 1/3, 2/3]), values[0]]
    else:
        return values
    
def get_cells(Gdata):
    cells = Gdata.ctx['cells']
    if Gdata.ctx['basis_type'] == 'gkhybrid': # phase space data
        if len(cells) == 5:
            return cells
        elif len(cells) == 4:
            return [cells[0], 2, cells[1], cells[2], cells[3]]
        elif len(cells) == 3:
            return [2, 2, cells[0], cells[1], cells[2]]
    elif Gdata.ctx['basis_type'] == 'serendipity': # configuration space
        if len(cells) == 3:
            return cells
        elif len(cells) == 2:
            return [cells[0], 2, cells[1]]
        elif len(cells) == 1:
            return [2, 2, cells[0]]
    else:
        return ValueError("Invalid basis type: %s"%Gdata.ctx['basis_type'])

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