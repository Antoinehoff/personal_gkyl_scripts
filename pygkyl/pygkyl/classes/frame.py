import postgkyl as pg
import numpy as np
from ..tools import math_tools as mt
import copy
from ..tools import pgkyl_interface as pgkyl_

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
    - fourrier_y: Applies FFT along the y dimension and updates the frame data.
    - info: Prints out key information about the frame.

    """
    def __init__(self, simulation, name, tf, load=False, polyorder=1, polytype='ms'):
        """
        Initialize a Frame instance with all attributes set to None.
        """
        self.simulation = simulation
        self.name = name
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
        self.tsymbol = None
        self.tunits = None
        self.Gdata = []
        self.dims = None
        self.ndims = None
        self.cells = None
        self.grids = None
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
        self.sliceddim = []
        self.slicecoords = {}
        self.slicetitle = ''
        self.timetitle = ''

        if load:
            self.load(polyorder=polyorder, polytype=polytype)

    def process_field_name(self):
        """
        Process the field name and set up the composition and filenames.
        """
        self.composition = self.simulation.normalization[self.name + 'compo']
        self.receipe = self.simulation.normalization[self.name + 'receipe']
        for subname in self.composition:
            subdataname = self.simulation.data_param.data_files_dict[subname + 'file']
            self.datanames.append(subdataname)
            if subname in ['b_x', 'b_y', 'b_z', 'Jacobian', 'Bmag']:
                name_tf = subdataname
            else:
                name_tf = '%s_%d' % (subdataname, self.tf)
            self.filenames.append("%s-%s.gkyl" % (self.simulation.data_param.fileprefix, name_tf))
            self.comp.append(self.simulation.data_param.data_files_dict[subname + 'comp'])
            self.gnames = copy.deepcopy(self.simulation.data_param.data_files_dict[subname + 'gnames'])
        self.gsymbols = [self.simulation.normalization[key + 'symbol'] for key in self.gnames]
        self.gunits = [self.simulation.normalization[key + 'units'] for key in self.gnames]
        self.vsymbol = self.simulation.normalization[self.name + 'symbol']
        self.vunits = self.simulation.normalization[self.name + 'units']

    def load(self, polyorder=1, polytype='ms'):
        """
        Load the data from the file and interpolate it.
        """
        self.Gdata = []
        for (f_, c_) in zip(self.filenames, self.comp):
            Gdata = pg.data.GData(f_)
            dg = pg.data.GInterpModal(Gdata, poly_order=polyorder, basis_type=polytype, periodic=False)
            dg.interpolate(c_, overwrite=True)
            self.Gdata.append(Gdata)
            if Gdata.ctx['time']:
                self.time = Gdata.ctx['time']
        self.grids = [g for g in pgkyl_.get_grid(Gdata) if len(g) > 2]
        self.cells = Gdata.ctx['cells']
        self.ndims = len(self.cells)
        self.dim_idx = list(range(self.ndims))
        if not self.time:
            self.time = 0
        self.refresh()
        self.normalize()

    def load_DG(self, polyorder=1, polytype='ms'):
        """
        Load the data from the file without interpolation.
        """
        self.Gdata = []
        Gdata = pg.data.GData(self.filenames[0])
        self.Gdata.append(Gdata)
        if Gdata.ctx['time']:
            self.time = Gdata.ctx['time']
        self.grids = [g for g in pgkyl_.get_grid(Gdata) if len(g) > 2]
        self.cells = Gdata.ctx['cells']
        self.ndims = len(self.cells)
        self.dim_idx = list(range(self.ndims))
        if not self.time:
            self.time = 0
        self.values = pgkyl_.get_values(Gdata)

    def refresh(self, values=True):
        """
        Refresh the grids and values.
        """
        self.new_cells = pgkyl_.get_cells(self.Gdata[0])
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
            self.time /= self.simulation.normalization['tscale']
            self.tsymbol = self.simulation.normalization['tsymbol']
            self.tunits = self.simulation.normalization['tunits']
        if grid:
            for ig in range(len(self.grids)):
                self.grids[ig] /= self.simulation.normalization[self.gnames[ig] + 'scale']
                self.grids[ig] -= self.simulation.normalization[self.gnames[ig] + 'shift']
                self.gsymbols[ig] = self.simulation.normalization[self.gnames[ig] + 'symbol']
                self.gunits[ig] = self.simulation.normalization[self.gnames[ig] + 'units']
        if values:
            self.values /= self.simulation.normalization[self.name + 'scale']
            self.values -= self.simulation.normalization[self.name + 'shift']
            self.vsymbol = self.simulation.normalization[self.name + 'symbol']
            self.vunits = self.simulation.normalization[self.name + 'units']

    def rename(self):
        """
        Rename the slice title and time title.
        """
        slicetitle = ''
        norm = self.simulation.normalization
        for k_, c_ in self.slicecoords.items():
            if isinstance(c_, float):
                slicetitle += norm[k_ + 'symbol'] + '=%3.3f' % c_ + norm[k_ + 'units'] + ', '
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

        if cut in ['avg', 'int']:
            cut_coord = direction + '-' + cut
            grid = self.simulation.geom_param.grids[ic][:]
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
        self.rename()

    def slice_1D(self, cutdirection, ccoords):
        """
        Slice the data to 1D along the specified direction and coordinates.
        """
        axes = 'xyz'.replace(cutdirection, '')
        for i_ in range(len(axes)):
            self.select_slice(direction=axes[i_], cut=ccoords[i_])
        self.refresh(values=False)

    def slice_2D(self, plane, ccoord):
        """
        Slice the data to 2D along the specified plane and coordinate.
        """
        i1 = getgrid_index(plane[0])
        i2 = getgrid_index(plane[1])
        i3 = 2 * (i1 == 0 and i2 == 1) + 1 * (i1 == 0 and i2 == 2) + 0 * (i1 == 1 and i2 == 2)
        sdir = self.gnames[i3]
        self.select_slice(direction=sdir, cut=ccoord)
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

        self.slice_2D(plane=direction, ccoord=ccoord)
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

    def fourrier_y(self):
        """
        Apply FFT along the y dimension and update the frame data.
        """
        fft_ky = np.fft.rfft(self.values, axis=1)
        Ny = self.values.shape[1]
        y = self.grids[1]
        dy = (y[1] - y[0]) * self.simulation.normalization['yscale']
        ky = np.fft.rfftfreq(Ny, d=dy)
        Nky = len(ky)
        ky = mt.create_uniform_array(ky, Nky + 1)
        ky = ky[1:]
        fft_ky = fft_ky[:, 1:, :]

        self.values = np.abs(fft_ky)
        gname = 'ky'
        self.grids[1] = ky
        self.gnames[1] = gname
        self.gsymbols[1] = self.simulation.normalization[gname + 'symbol']
        self.gunits[1] = self.simulation.normalization[gname + 'units']

        self.refresh(values=False)

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