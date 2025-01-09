# This file contains functions that are used to interface with the postgkyl library.
import postgkyl as pg
import numpy as np

def get_values(Gdata):
    if Gdata.get_values().ndim == 3 :
        values = np.expand_dims(Gdata.get_values(), axis=1)
        return np.concatenate((values, values), axis=1)
    else:
        return Gdata.get_values()
    
def interpolate(dg,comp,overwrite):
    values = interpolate(comp,overwrite)
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