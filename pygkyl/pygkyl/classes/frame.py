import postgkyl as pg
import numpy as np
from ..tools import math_tools as mt
import copy
from ..interfaces import pgkyl_interface as pgkyl_
from ..interfaces import flaninterface as flan_
from ..tools import DG_tools, fig_tools

def getgrid_index(s):
    return (1*(s == 'x') + 2*(s == 'y') + 3*(s == 'z') + 4*(s == 'v') + 5*(s == 'm')) - 1

class Frame:
    """
    Frame
    -----
    Manages the loading, slicing, and normalization of simulation data frames.

    Methods:
    --------
    - __init__: Initializes the Frame object with the required parameters.
    - process_field_name: Processes the field name and sets up the composition and filenames.
    - load: Loads the data from the file and interpolates it.
    - load_DG: Loads the data from the file without interpolation.
    - refresh: Refreshes the grids and values.
    - normalize: Normalizes the time, grids, and values.
    - rename: Renames the slice title and time title.
    - select_slice: Selects a slice of the data along a specified direction.
    - slice_1D: Slices the data to 1D along the specified direction and coordinates.
    - slice_2D: Slices the data to 2D along the specified plane and coordinate.
    - compute_volume_integral: Computes the volume integral of the data.
    - compute_surface_integral: Computes the surface integral of the data.
    - free_values: Frees the values to save memory.
    - fourier_y: Applies FFT along the y dimension and updates the frame data.
    - info: Prints out key information about the frame.

    """
    def __init__(self, simulation, fieldname, tf, load=False, fourier_y=False,
                 polyorder=1, polytype='ms', normalize=True):
        """
        Initialize a Frame instance with all attributes set to None.
        """
        self.simulation = simulation
        self.name = fieldname
        self.tf = tf
        self.datanames = []
        self.filenames = []
        self.comp = []
        self.gnames = None
        self.gsymbols = None
        self.gunits = None
        self.vsymbol = None
        self.vunits = None
        self.composition = []
        self.process_field_name()  # this initializes the above attributes

        self.time = None
        self.tsymbol = ''
        self.tunits = ''
        self.Gdata = []
        self.dims = None
        self.ndims = None
        self.cells = None
        self.grids = None # physical grids (get normalized)
        self.cgrids = None # computational grids
        self.values = None
        self.Jacobian = simulation.geom_param.Jacobian
        self.vol_int = None
        self.vol_avg = None

        # attribute to handle slices
        self.dim_idx = None
        self.new_grids = []
        self.new_gnames = []
        self.new_gsymbols = []
        self.new_gunits = []
        self.new_dims = []
        self.mgrids = []
        self.sliceddim = []
        self.slicecoords = {}
        self.sliceindex = {}
        self.slicetitle = ''
        self.timetitle = ''
        self.fulltitle = ''
        
        if simulation.code == 'gyacomo':
            self.load = self.load_gyac
            self.get_cells = self.get_cells_gyac
        elif 'flan' in fieldname:
            self.load = self.load_flan
            self.get_cells = self.get_cells_flan
        else: # Gkeyll by default
            self.load = self.load_gkyl
            self.get_cells = self.get_cells_pgkyl

        if load:
            self.load(polyorder=polyorder, polytype=polytype, normalize=normalize, fourier_y=fourier_y)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.free_values()
    
    def process_field_name(self):
        """
        Process the field name and set up the composition and filenames.
        """
        try:
            self.composition = self.simulation.data_param.field_info_dict[self.name + 'compo']
        except KeyError as e:
            raise KeyError(f"Cannot find receipe for '{self.name}'. "
                           f"You can check available field names with simulation.data_param.info()")
        self.receipe = self.simulation.data_param.field_info_dict[self.name + 'receipe']
        for subname in self.composition:
            subdataname = self.simulation.data_param.file_info_dict[subname + 'file']
            self.datanames.append(subdataname)
            if subname in ['b_x', 'b_y', 'b_z', 'Jacobian', 'Bmag', 
                           'g_xx', 'g_xy', 'g_xz', 'g_yy', 'g_yz', 'g_zz',
                           'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']:
                name_tf = subdataname
            else:
                name_tf = '%s_%d' % (subdataname, self.tf)
            self.filenames.append("%s-%s.gkyl" % (self.simulation.data_param.fileprefix, name_tf))
            self.comp.append(self.simulation.data_param.file_info_dict[subname + 'comp'])
            self.gnames = copy.deepcopy(self.simulation.data_param.file_info_dict[subname + 'gnames'])
        self.gsymbols = [self.simulation.normalization.dict[key + 'symbol'] for key in self.gnames]
        self.gunits = [self.simulation.normalization.dict[key + 'units'] for key in self.gnames]
        self.vsymbol = self.simulation.normalization.dict[self.name + 'symbol']
        self.vunits = self.simulation.normalization.dict[self.name + 'units']

    def get_DG_coeff(self):
        """
        Get the DG coefficients.
        """
        Gdata_list = []
        for (f_, c_) in zip(self.filenames, self.comp):
            gdata_ = copy.deepcopy(pg.data.GData(f_))
            gdata_.values = gdata_.values[:,:,:,c_*8:(c_+1)*8]
            Gdata_list.append(gdata_)
        values = copy.deepcopy(self.receipe(Gdata_list))
        Gdata_out = copy.deepcopy(Gdata_list[0])
        Gdata_out.values = values
        return Gdata_out
    
    def eval_DG_proj(self, grid=None):
        DGbasis = DG_tools.DG_basis()
        DGdata = self.get_DG_coeff()
        
        if grid is None:
            grid = self.new_grids

        xg = [grid[0]] if isinstance(grid[0], float) else grid[0]
        yg = [grid[1]] if isinstance(grid[1], float) else grid[1]
        zg = [grid[2]] if isinstance(grid[2], float) else grid[2]
        
        projection = np.zeros((len(xg), len(yg), len(zg)))
        for i in range(len(xg)):
            for j in range(len(yg)):
                for k in range(len(zg)):
                    projection[i, j, k] = DGbasis.eval_proj(DGdata, [xg[i], yg[j], zg[k]])
                    
        return projection.squeeze()
    
    def load_gkyl(self, polyorder=1, polytype='ms', normalize=True, fourier_y=False):
        """
        Load the data from the file and interpolate it.
        """
        self.Gdata = []
        for (f_, c_) in zip(self.filenames, self.comp):
            Gdata = pg.data.GData(f_)
            dg = pg.data.GInterpModal(Gdata, poly_order=polyorder, basis_type=polytype, periodic=False)
            xNodal, _ = dg.interpolate(c_)
            self.xNodal = xNodal
            dg.interpolate(c_, overwrite=True)
            self.Gdata.append(Gdata)
            if Gdata.ctx['time']:
                self.time = Gdata.ctx['time']
                
        #.Centered mesh creation
        gridsN = self.xNodal # nodal grids
        mesh = []
        for i in range(self.simulation.DG_basis.dimensionality):
            nNodes  = len(gridsN[i])
            mesh.append(np.multiply(0.5,gridsN[i][0:nNodes-1]+gridsN[i][1:nNodes]))
        self.cgrids = [m for m in mesh]
        self.grids = [g for g in pgkyl_.get_grid(Gdata) if len(g) > 2]
        self.cells = Gdata.ctx['cells']
        self.ndims = len(self.cells)
        self.dim_idx = list(range(self.ndims))
        if not self.time: self.time = 0
        self.refresh()
        if normalize: self.normalize()
        if fourier_y: self.fourier_y()

    def get_cells_pgkyl(self):
        return pgkyl_.get_cells(self.Gdata[0])

    def load_flan(self, polyorder=1, polytype='ms', normalize=True, fourier_y=False):
        flan = flan_.FlanInterface(self.simulation.flandatapath)
        self.time, self.grids, self.Jacobian, self.values = flan.load_data(self.name, self.tf, xyz= not fourier_y)
        self.cgrids = [g for g in self.grids]
        if normalize: self.normalize(values=False)
        
    def get_cells_flan(self):
        return [len(grid) for grid in self.grids]
    
    def load_gyac(self, polyorder=1, polytype='ms', normalize=True, fourier_y=False):
        self.grids, self.time, self.values = self.simulation.gyac.load_data(self.name, self.tf, xyz= not fourier_y)
        _, _, _, symbols = self.simulation.gyac.field_map[self.name]
        self.gsymbols = [self.simulation.normalization.dict[key + 'symbol'] for key in self.gnames]
        self.gunits = [self.simulation.normalization.dict[key + 'units'] for key in self.gnames]
        self.vsymbol = self.simulation.normalization.dict[self.name + 'symbol']
        self.vunits = self.simulation.normalization.dict[self.name + 'units']
        self.Jacobian = np.ones_like(self.values)
        self.xNodal = self.grids.copy()
        self.cgrids = self.grids.copy()
        if normalize: self.normalize()
        self.refresh(values=False)
        
    def get_cells_gyac(self):
        return [len(grid) for grid in self.grids]

    def refresh(self, values=True):
        """
        Refresh the grids and values.
        """
        self.new_cells = self.get_cells()
        self.new_grids = []
        self.new_gnames = []
        self.new_gsymbols = []
        self.new_gunits = []
        self.new_dims = [c_ for c_ in self.new_cells if c_ > 1]
        self.dim_idx = [d_ for d_ in range(3) if d_ not in self.sliceddim]
        for idx in self.dim_idx:
            Ngidx = len(self.grids[idx])
            self.new_grids.append(mt.create_uniform_array(self.grids[idx], Ngidx - 1))
            self.new_gnames.append(self.gnames[idx])
            self.new_gsymbols.append(self.gsymbols[idx])
            self.new_gunits.append(self.gunits[idx])
        if values:
            self.values = copy.deepcopy(self.receipe(self.Gdata))
            self.values = self.values.reshape(self.new_dims)
            self.values = np.squeeze(self.values)

    def normalize(self, values=True, time=True, grid=True):
        """
        Normalize the time, grids, and values.
        """
        if time:
            self.time /= self.simulation.normalization.dict['tscale']
            self.tsymbol = self.simulation.normalization.dict['tsymbol']
            self.tunits = self.simulation.normalization.dict['tunits']
        if grid:
            for ig in range(len(self.grids)):
                self.grids[ig] /= self.simulation.normalization.dict[self.gnames[ig] + 'scale']
                self.grids[ig] -= self.simulation.normalization.dict[self.gnames[ig] + 'shift']
                self.gsymbols[ig] = self.simulation.normalization.dict[self.gnames[ig] + 'symbol']
                self.gunits[ig] = self.simulation.normalization.dict[self.gnames[ig] + 'units']
        if values:
            self.values /= self.simulation.normalization.dict[self.name + 'scale']
            self.values -= self.simulation.normalization.dict[self.name + 'shift']
            self.vsymbol = self.simulation.normalization.dict[self.name + 'symbol']
            self.vunits = self.simulation.normalization.dict[self.name + 'units']

    def refresh_title(self):
        """
        Refresh the slice title and time title.
        """
        slicetitle = ''
        norm = self.simulation.normalization.dict
        for k_, c_ in self.slicecoords.items():
            if isinstance(c_, float):
                fmt = fig_tools.optimize_str_format(c_)
                slicetitle += norm[k_ + 'symbol'] + '=' + fmt%c_ + norm[k_ + 'units'] + ', '
            else:
                slicetitle += c_ + ', '

        self.slicetitle = slicetitle
        self.timetitle = self.tsymbol + '=%2.2f' % self.time + self.tunits
        self.fulltitle = self.slicetitle + self.timetitle

    def select_slice(self, direction, cut):
        """
        Select a slice of the data along a specified direction.
        """
        direction_map = {'x': 0, 'y': 1, 'z': 2, 'vpar': 3, 'mu': 4}

        if direction not in direction_map:
            raise ValueError("Invalid direction '" + direction + "': must be 'x', 'y', 'z', 'vpar', or 'mu'")

        ic = direction_map[direction]

        cut_index = -1
        if cut in ['avg', 'int']:
            cut_coord = direction + '-' + cut
            grid = self.cgrids[ic][:]
            self.values = np.trapz(self.values * self.Jacobian, grid, axis=ic)
            self.Jacobian = np.trapz(self.Jacobian, grid, axis=ic)
            if cut == 'avg':
                self.values /= self.Jacobian
        elif cut == 'max':
            cut_coord = direction + '-max'
            self.values = np.max(self.values, axis=ic)
        elif cut == 'mean':
            cut_coord = direction + '-mean'
            self.values = np.mean(self.values, axis=ic)
        else:
            if isinstance(cut, int):
                cut_index = np.minimum(cut, len(self.grids[ic]) - 2)
            else:
                cut_index = (np.abs(self.grids[ic] - cut)).argmin()
                cut_index = np.minimum(cut_index, len(self.grids[ic]) - 2)
            cut_coord = self.grids[ic][cut_index]
            self.values = np.take(np.copy(self.values), cut_index, axis=ic)
            self.Jacobian = np.take(np.copy(self.Jacobian), cut_index, axis=ic)

        self.values = np.expand_dims(self.values, axis=ic)
        self.Jacobian = np.expand_dims(self.Jacobian, axis=ic)

        self.sliceddim.append(ic)
        self.slicecoords[direction] = cut_coord
        self.sliceindex[direction] = cut_index
        self.refresh_title()

    def slice(self, axs, ccoord):
        """
        Slice the data along specified axes and coordinates.
        1. axs: string of axes to slice along (e.g., 'xy', 'yz', 'xz', 'scalar')
        2. ccoord: list of coordinates to slice at (e.g., [x, y, z])
        """
        ccoord = [ccoord] if not isinstance(ccoord, list) else ccoord
        
        if axs in ['fluxsurf','phitheta', 'fs']:
            self.flux_surface_projection(xcut=ccoord[0])
        else:
            ax_to_cut = 'xyz'
            if axs == 'scalar':
                ax_to_cut = 'xyz'
            else:
                for i_ in range(len(axs)): ax_to_cut = ax_to_cut.replace(axs[i_], '')
            for i_ in range(len(ax_to_cut)):
                self.select_slice(direction=ax_to_cut[i_], cut=ccoord[i_])
        self.refresh(values=False)
        
    def compute_volume_integral(self, jacob_squared=False, average=False):
        """
        Compute the volume integral of the data.
        """
        [x, y, z] = self.simulation.geom_param.grids
        Jac = self.simulation.geom_param.Jacobian**2 if jacob_squared else self.simulation.geom_param.Jacobian
        self.vol_int = mt.integral_vol(x, y, z, self.values * Jac)
        if average:
            self.vol_int /= self.simulation.geom_param.intJac
        return self.vol_int

    def compute_surface_integral(self, direction='yz', ccoord=0, integrant_filter="all",
                                 int_bounds=['all', 'all'], surf_coord='all'):
        """
        Compute the surface integral of the data.

        Parameters:
        - direction: string of axes to slice along (e.g., 'xy', 'yz', 'xz')
        - ccoord: list of coordinates to slice at (e.g., [x, y, z])
        - integrant_filter: string to filter the integrant ('all', 'pos', 'neg')
        - int_bounds: list of bounds for the integration (e.g., [[x1, x2], [y1, y2]])
        - surf_coord: coordinate for the surface integral (e.g., 'x', 'y', 'z')
        """
        [x, y, z] = self.simulation.geom_param.grids
        dir_dict = {'x': [0, x], 'y': [1, y], 'z': [2, z]}

        if len(direction) != 2 or any(d not in dir_dict for d in direction):
            raise ValueError("Direction must be a two-character string from 'x', 'y', 'z'")

        [dir1, grid1] = dir_dict[direction[0]]
        [dir2, grid2] = dir_dict[direction[1]]
        il1, iu1 = 0, len(grid1)
        il2, iu2 = 0, len(grid2)

        if isinstance(int_bounds[0], list):
            il1 = np.argmin(np.abs(grid1 - int_bounds[0][0]))
            iu1 = np.argmin(np.abs(grid1 - int_bounds[0][1]))
        if isinstance(int_bounds[1], list):
            il2 = np.argmin(np.abs(grid2 - int_bounds[1][0]))
            iu2 = np.argmin(np.abs(grid2 - int_bounds[1][1]))
        
        self.slice(axs=direction, ccoord=ccoord)

        integrant = self.values * self.Jacobian

        slices = [slice(None)] * 3
        for bound, dir in zip([[il1, iu1], [il2, iu2]], [dir1, dir2]):
            slices[dir] = slice(None, bound[0])
            integrant[tuple(slices)] = 0
            slices[dir] = slice(bound[1] + 1, None)
            integrant[tuple(slices)] = 0
            slices[dir] = slice(None)

        if integrant_filter == "pos":
            integrant[integrant < 0.0] = 0.0
        elif integrant_filter == "neg":
            integrant[integrant > 0.0] = 0.0

        surf_int_z = np.trapz(integrant, x=grid1, axis=dir1)
        surf_int_z = np.expand_dims(surf_int_z, axis=dir1)
        surf_int = np.trapz(surf_int_z, x=grid2, axis=dir2)
        self.surf_int = surf_int.squeeze()

        return self.surf_int

    def free_values(self):
        """
        Free the values to save memory.
        """
        self.values = None

    def fourier_y(self):
        """
        Apply FFT along the y dimension and update the frame data.
        """
        fft_ky = np.fft.rfft(self.values, axis=1)
        Ny = self.values.shape[1]
        y = self.grids[1]
        dy = (y[1] - y[0]) * self.simulation.normalization.dict['yscale']
        ky = np.fft.rfftfreq(Ny, d=dy)
        Nky = len(ky)
        ky = mt.create_uniform_array(ky, Nky + 1)
        ky = ky[1:]
        fft_ky = fft_ky[:, 1:, :]

        self.values = np.abs(fft_ky)
        gname = 'ky'
        self.grids[1] = ky/self.simulation.normalization.dict[gname + 'scale']
        self.gnames[1] = gname
        self.gsymbols[1] = self.simulation.normalization.dict[gname + 'symbol']
        self.gunits[1] = self.simulation.normalization.dict[gname + 'units']
        
        # We average the jacobian in the y direction
        self.Jacobian = np.mean(self.Jacobian, axis=1)
        # And replicate it to match the new shape of values
        self.Jacobian = np.repeat(self.Jacobian[:, np.newaxis, :],
                                  self.values.shape[1], axis=1)

        self.refresh(values=False)
        
    def copy(self):
        """
        Copy the frame.
        """
        return copy.deepcopy(self)

    def info(self):
        """
        Print out key information about the frame.
        """
        print(f"Frame Name: {self.name}")
        print(f"Simulation: {self.simulation}")
        print(f"Time Frame: {self.tf}")
        print(f"Time: {self.time}")
        print(f"Dimensions: {self.ndims}")
        print(f"Grid Names: {self.gnames}")
        print(f"Grid Symbols: {self.gsymbols}")
        print(f"Grid Units: {self.gunits}")
        print(f"Value Symbol: {self.vsymbol}")
        print(f"Value Units: {self.vunits}")
        print(f"Composition: {self.composition}")
        print(f"Data Names: {self.datanames}")
        print(f"File Names: {self.filenames}")
        print(f"Jacobian: {self.Jacobian}")
        print(f"Volume Integral: {self.vol_int}")
        print(f"Surface Integral: {self.surf_int}")
        print(f"Sliced Dimensions: {self.sliceddim}")
        print(f"Slice Coordinates: {self.slicecoords}")
        print(f"Slice Title: {self.slicetitle}")
        print(f"Time Title: {self.timetitle}")
        print(f"Full Title: {self.fulltitle}")