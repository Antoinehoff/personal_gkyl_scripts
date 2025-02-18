# Set of routines to handle the DG representation of fields
import numpy as np
import matplotlib.pyplot as plt

class DG_basis: 
    order = 1
    type = 'ser'
    dimensionality = 3
    simulation = None

    def __init__(self,order=1,type='ser',dimensionality=3):
        self.order = order
        self.type = type
        if dimensionality in [3,'3x2v']:
            self.dimensionality = 3
        elif dimensionality in [2,'2x2v']:
            self.dimensionality = 2
        else:
            raise Exception("Supported dimensionality is 3 or 2")

        if(self.order != 1 or self.type != 'ser'):
            raise Exception("Only linear order serendipity basis supported")

        self.psi_i = None
        self.init_psi()

    def init_psi(self):
        '''
        Initialize the basis functions
        '''
        if self.dimensionality == 3:
            coeff = 1./2.**(1.5)
            def psi_i(i,x,y,z):
                if  (i == 0): return coeff
                elif(i == 1): return coeff * np.sqrt(3) * x
                elif(i == 2): return coeff * np.sqrt(3) * y
                elif(i == 3): return coeff * np.sqrt(3) * z
                elif(i == 4): return coeff * 3 * x * y
                elif(i == 5): return coeff * 3 * x * z
                elif(i == 6): return coeff * 3 * y * z
                elif(i == 7): return coeff * 3**(1.5) * x * y * z
                else: raise Exception("Invalid basis function index")
            def gradpsi_i (i,x,y,z,j):
                if  (i == 0): return 0.0
                elif(i == 1): return coeff * np.sqrt(3) if j == 0 else 0.0
                elif(i == 2): return coeff * np.sqrt(3) if j == 1 else 0.0
                elif(i == 3): return coeff * np.sqrt(3) if j == 2 else 0.0
                elif(i == 4): return coeff * 3 * (y * (j == 0) + x * (j == 1))
                elif(i == 5): return coeff * 3 * (z * (j == 0) + x * (j == 2))
                elif(i == 6): return coeff * 3 * (z * (j == 1) + y * (j == 2))
                elif(i == 7): return coeff * 3**(1.5) * (y * z * (j == 0) + x * z * (j == 1) + x * y * (j == 2))
                else: raise Exception("Invalid basis function index")
        elif self.dimensionality == 2: 
            coeff = 1./2.
            def psi_i(i,x,y):
                if  (i == 0): return coeff
                elif(i == 1): return coeff * np.sqrt(3) * x
                elif(i == 2): return coeff * np.sqrt(3) * y
                elif(i == 3): return coeff * 3 * x * y
                else: raise Exception("Invalid basis function index")
            def gradpsi_i (i,x,y,j):
                if  (i == 0): return 0.0
                elif(i == 1): return coeff * np.sqrt(3) if j == 0 else 0.0
                elif(i == 2): return coeff * np.sqrt(3) if j == 1 else 0.0
                elif(i == 3): return coeff * 3 * (y * (j == 0) + x * (j == 1))
                else: raise Exception("Invalid basis function index")
        else: raise Exception("Invalid dimensionality")
        
        self.psi_i = psi_i
        self.gradpsi_i = gradpsi_i

    def eval_proj(self, Gdata, coords, id = None):
        '''
        Evaluate the projection of the DG field at the point x,y,z

        Args:
            Gdata (GridData): Grid data object
            coords (list): list of coordinates [x,y,z]
            id (int): index of the component to evaluate the gradient
        '''
        val = 0.0

        ix,iy,iz, xc,yc,zc, gradc = self.grid_to_cell(Gdata, coords)

        if id is None:
            for i in range(8):
                val += self.psi_i(i,xc,yc,zc)*Gdata.values[ix,iy,iz,i]
        else:
            for i in range(8):
                val += gradc[id]*self.gradpsi_i(i,xc,yc,zc,j=id)*Gdata.values[ix,iy,iz,i]
        
        return val
    
    def grid_to_cell(self, Gdata, coords):
        xnodes = Gdata.grid[0]
        ynodes = Gdata.grid[1]
        znodes = Gdata.grid[2]
        x = coords[0]
        y = coords[1]
        z = coords[2]
        
        gradc = np.zeros(3)

        ix = np.argmin(np.abs(xnodes-x))
        if( x < xnodes[ix] or x >= max(xnodes) ): ix -= 1
        x0 = xnodes[ix]
        x1 = xnodes[ix+1]
        xc = 2*(x-x0)/(x1-x0) - 1
        gradc[0] = 2/(x1-x0)

        iy = np.argmin(np.abs(ynodes-y))
        if( y < ynodes[iy] or y >= max(ynodes)): iy -= 1
        y0 = ynodes[iy]
        y1 = ynodes[iy+1]
        yc = 2*(y-y0)/(y1-y0) - 1
        gradc[1] = 2/(y1-y0)

        iz = np.argmin(np.abs(znodes-z))
        if( z < znodes[iz] or z >= max(znodes) ): iz -= 1
        z0 = znodes[iz]
        z1 = znodes[iz+1]
        zc = 2*(z-z0)/(z1-z0) - 1
        gradc[2] = 2/(z1-z0)

        return ix,iy,iz, xc,yc,zc, gradc
    
    def info(self):
        '''
        Print the attributes
        '''
        print('order = ',self.order)
        print('type = ',self.type)
        print('dimensionality = ',self.dimensionality)