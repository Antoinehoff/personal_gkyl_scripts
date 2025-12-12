# Set of routines to handle the DG representation of fields
import numpy as np
from . import gkhyb_basis as bf

class DGBasis: 
    order = 1
    type = 'ms'
    ndim = 3
    simulation = None
    psi_i = None
    gradpsi_i = None

    def __init__(self,ndim=3):
        self.ndim = ndim
        self.psi_i = None
        self.init_psi()

    def init_psi(self):
        '''
        Initialize the basis functions
        '''
        if self.ndim == 5:
            # 3x2v: configuration space is 3D (x,y,z)
            self.psi_i = bf.gkhyb_3x2v_phase
            self.gradpsi_i = bf.grad_gkhyb_3x2v_phase
            self.num_basis = bf.num_basis_3x2v_phase
        elif self.ndim == 4:
            # 2x2v: configuration space is 2D (x,z)
            self.psi_i = bf.gkhyb_2x2v_phase
            self.gradpsi_i = bf.grad_gkhyb_2x2v_phase
            self.num_basis = bf.num_basis_2x2v_phase
        elif self.ndim == 3:
            # 3x: configuration space is 3D (x,y,z)
            self.psi_i = bf.gkhyb_3x_conf
            self.gradpsi_i = bf.grad_gkhyb_3x_conf
            self.num_basis = bf.num_basis_3x_conf
        elif self.ndim == 2: 
            # 2x: configuration space is 2D (x,z)
            self.psi_i = bf.gkhyb_2x_conf
            self.gradpsi_i = bf.grad_gkhyb_2x_conf
            self.num_basis = bf.num_basis_2x_conf
        elif self.ndim == 1:
            # 1x: configuration space is 1D (x)
            self.psi_i = bf.gkhyb_1x_conf
            self.gradpsi_i = bf.grad_gkhyb_1x_conf
            self.num_basis = bf.num_basis_1x_conf
                
    def eval_proj(self, Gdata, coords, id = None):
        '''
        Evaluate the projection of the DG field at the point coords

        Args:
            Gdata : Gkeyll data object
            coords (list): list of coordinates
            id (int): index of the component to evaluate the gradient
        '''
        cell_idx, cell_coord, gradc = self.grid_to_cell(Gdata, coords)

        val = 0.0
        if id is None:
            for i in range(self.num_basis()):
                val += self.psi_i(i,cell_coord)*self.get_coeffs(Gdata,cell_idx,i)
        else:
            for i in range(self.num_basis()):
                # gradpsi_i returns an array, index with [id] to get the specific component
                val += gradc[id]*self.gradpsi_i(i,cell_coord)[id]*self.get_coeffs(Gdata,cell_idx,i)

        return val
    
    def grid_to_cell(self, Gdata, coords):
        '''
        Map global coordinates to cell index and local coordinates in cell
        '''
        ndim = len(Gdata.grid)
        gradc = np.zeros(ndim)
        cell_idx = np.zeros(ndim,dtype=int)
        cell_coord = np.zeros(ndim)
        cell_grad = np.zeros(ndim)
        
        for i in range(ndim):
            nodes = Gdata.grid[i]
            x = coords[i]
            if isinstance(x, int):
                ix = x
                x = nodes[ix]
            else:
                ix = np.argmin(np.abs(nodes-x))
                
            if x > max(nodes):
                ix = -1
                xc = 1.0
                gradc = 2/(nodes[-1]-nodes[-2])
            elif x < min(nodes):
                ix = 0
                xc = -1.0
                gradc = 2/(nodes[1]-nodes[0])
            else:
                if( x < nodes[ix] or x >= max(nodes) ): ix -= 1
                x0 = nodes[ix]
                x1 = nodes[ix+1]
                xc = 2*(x-x0)/(x1-x0) - 1
                xc = min(1.0, max(-1.0, xc)) # clamp to [-1,1]
                gradc = 2/(x1-x0)
            cell_grad[i] = gradc
            cell_idx[i] = ix
            cell_coord[i] = xc

        return cell_idx, cell_coord, cell_grad
    
    def get_coeffs(self, Gdata, idx, i):
        return Gdata.values[tuple(idx)][i]

    def info(self):
        '''
        Print the attributes
        '''
        print('order = ',self.order)
        print('ndim = ',self.ndim)