import netCDF4
import os
import numpy as np

class FlanInterface:
    
    def __init__(self,path):
        self.path = path
        if self.path is None:
            self.avail_frames = []
        elif not os.path.exists(self.path):
            raise FileNotFoundError(f"Flan data file {self.path} not found.") 
        else:      
            flan_nc = netCDF4.Dataset(self.path, 'r')
            self.avail_frames = [i for i in range(len(flan_nc.variables['time'][:].data))]
        
    def load_data(self, fieldname, tframe, xyz=True):
        dname = fieldname.replace('flan_', '')
        flan_nc = netCDF4.Dataset(self.path, 'r')
        values = flan_nc[dname][tframe].data
        time = float(flan_nc.variables['time'][tframe].data)
        # get x grid and convert to default dtype instead of float32
        xc = flan_nc.variables['x'][:].data
        yc = flan_nc.variables['y'][:].data
        zc = flan_nc.variables['z'][:].data
        # evaluate grid at nodal points (i.e. add a point at the end of the grid)
        x = np.append(xc, xc[-1] + (xc[-1] - xc[-2]))
        y = yc
        z = zc
        
        grids = [x.astype(float), y.astype(float), z.astype(float)]
        
        jacobian = flan_nc['J'][:].data.astype(float)
        
        jacobian = 0.5* (jacobian[1:,1:,1:] + jacobian[:-1,:-1,:-1])
        
        return time, grids, jacobian, values
