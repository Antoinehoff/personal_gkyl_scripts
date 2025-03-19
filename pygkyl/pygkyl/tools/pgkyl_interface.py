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

def get_values(Gdata):
    # handle 2D and 3D data differently
    if Gdata.get_values().ndim == 3 :
        values = np.expand_dims(Gdata.get_values(), axis=1)
        return np.concatenate((values, values), axis=1)
    else:
        return Gdata.get_values()

def interpolate(Gdata,comp,polyorder=1, polytype='ms'):
    dg = pg.data.GInterpModal(Gdata, poly_order=polyorder, basis_type=polytype, periodic=False)
    values = dg.interpolate(comp)
    if values.ndim == 3:
        return values
    if values.ndim == 2:
        values = np.expand_dims(values, axis=1)
        return np.concatenate((values, values), axis=1)
    
def get_grid(Gdata):
    values = Gdata.get_grid()
    if len(values) == 3 :
        return values
    elif len(values) == 2 :
        return [values[0], np.array([0, 1/3, 2/3]), values[1]]
    
def get_cells(Gdata):
    cells = Gdata.ctx['cells']
    if len(cells) == 3 :   
        return cells
    else :        
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