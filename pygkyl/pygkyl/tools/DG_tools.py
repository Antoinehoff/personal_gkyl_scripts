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
                if(i == 0):
                    return coeff
                elif(i == 1):
                    return coeff * np.sqrt(3) * x
                elif(i == 2):
                    return coeff * np.sqrt(3) * y
                elif(i == 3):
                    return coeff * np.sqrt(3) * z
                elif(i == 4):
                    return coeff * 3 * x * y
                elif(i == 5):
                    return coeff * 3 * x * z
                elif(i == 6):
                    return coeff * 3 * y * z
                elif(i == 7):
                    return coeff * 3**(1.5) * x * y * z
                else:
                    raise Exception("Invalid basis function index")
            self.psi_i = psi_i
        elif self.dimensionality == 2: 
            coeff = 1./2.
            def psi_i(i,x,y):
                if(i == 0):
                    return coeff
                elif(i == 1):
                    return coeff * np.sqrt(3) * x
                elif(i == 2):
                    return coeff * np.sqrt(3) * y
                elif(i == 3):
                    return coeff * 3 * x * y
                else:
                    raise Exception("Invalid basis function index")
            self.psi_i = psi_i

    def eval_proj(self, Gdata, coords):
        '''
        Evaluate the projection of the DG field at the point x,y,z
        '''
        val = 0.0
        xnodes = Gdata.grid[0]
        ynodes = Gdata.grid[1]
        znodes = Gdata.grid[2]
        x = coords[0]
        y = coords[1]
        z = coords[2]

        ix = np.argmin(np.abs(xnodes-x))
        if( x < xnodes[ix] or x >= max(xnodes) ): ix -= 1
        x0 = xnodes[ix]
        x1 = xnodes[ix+1]
        xc = 2*(x-x0)/(x1-x0) - 1

        iy = np.argmin(np.abs(ynodes-y))
        if( y < ynodes[iy] or y >= max(ynodes)): iy -= 1
        y0 = ynodes[iy]
        y1 = ynodes[iy+1]
        yc = 2*(y-y0)/(y1-y0) - 1

        iz = np.argmin(np.abs(znodes-z))
        if( z < znodes[iz] or z >= max(znodes) ): iz -= 1
        z0 = znodes[iz]
        z1 = znodes[iz+1]
        zc = 2*(z-z0)/(z1-z0) - 1

        for i in range(8):
            val += self.psi_i(i,xc,yc,zc)*Gdata.values[ix,iy,iz,i]
        
        return val

    def eval_on_grid(self, Gdata, xgrid, ygrid, zgrid):
        '''
        Evaluate the projection of the DG field on the grid
        '''
        # first check if grids are not scalar
        if np.isscalar(xgrid): xgrid = np.array([xgrid])
        if np.isscalar(ygrid): ygrid = np.array([ygrid])
        if np.isscalar(zgrid): zgrid = np.array([zgrid])

        nx = len(xgrid)
        ny = len(ygrid)
        nz = len(zgrid)
        out = np.zeros((nx,ny,nz))
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    x = xgrid[ix]
                    y = ygrid[iy]
                    z = zgrid[iz]
                    out[ix,iy,iz] = self.eval_proj(Gdata,[x,y,z])
        return np.squeeze(out)
    
    def plot_proj(self, Gdata, direction='x', coords = [0.0, 0.0], xlim = [], show_cells = False):
        '''
        Plot the projection of the DG field on the grid
        '''
        if self.dimensionality != 3:
            raise Exception("Only 3D fields can be plotted")

        if direction == 'x':
            cells = Gdata.grid[0]
            c0 = coords[0] + 0.5*(Gdata.grid[1][1]-Gdata.grid[1][0])
            c1 = coords[1] + 0.5*(Gdata.grid[2][1]-Gdata.grid[2][0])
            def coord_swap(x): return [x,c0,c1]
        elif direction == 'y':
            cells = Gdata.grid[1]
            c0 = coords[0] + 0.5*(Gdata.grid[0][1]-Gdata.grid[0][0])
            c1 = coords[1] + 0.5*(Gdata.grid[2][1]-Gdata.grid[2][0])
            def coord_swap(y): return [c0,y,c1]
        elif direction == 'z':
            cells = Gdata.grid[2]
            c0 = coords[0] + 0.5*(Gdata.grid[0][1]-Gdata.grid[0][0])
            c1 = coords[1] + 0.5*(Gdata.grid[1][1]-Gdata.grid[1][0])
            def coord_swap(z): return [c0,c1,z]
        else:
            raise Exception("Invalid direction")
        
        dx = cells[1]-cells[0]
        DG_proj = []
        x_proj  = []
        for ic in range(len(cells)-1):
            xi = cells[ic]+0.01*dx
            fi = self.eval_proj(Gdata, coord_swap(xi))
            xip1 = cells[ic]+0.99*dx
            fip1 = self.eval_proj(Gdata, coord_swap(xip1))
            DG_proj.append(fi)
            x_proj.append(xi)
            DG_proj.append(fip1)
            x_proj.append(xip1)
            DG_proj.append(None)
            x_proj.append(cells[ic])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_proj, DG_proj,'-')
        # add a vertical line to mark each cell boundary
        if show_cells:
            for xc in cells:
                    ax.axvline(xc, color='k', linestyle='-', alpha=0.25)
        if xlim: ax.set_xlim(xlim)
        plt.show()