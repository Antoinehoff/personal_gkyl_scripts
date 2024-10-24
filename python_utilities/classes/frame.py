import postgkyl as pg
import numpy as np
from tools import math_tools as mt
import copy
def getgrid_index(s):
    return  (1*(s == 'x') + 2*(s == 'y') + 3*(s == 'z') + 4*(s == 'v') + 5*(s == 'm'))-1

class Frame:
    def __init__(self,simulation,name,tf,load=False,polyorder=1,polytype='ms'):
        """
        Initialize a Frame instance with all attributes set to None.
        """
        self.simulation     = simulation
        self.name           = name
        self.tf             = tf
        self.datanames       = []
        self.filenames       = []
        self.comp           = []
        self.gnames         = None
        self.gsymbols       = None
        self.gunits         = None
        self.vsymbol        = None
        self.vunits         = None
        self.composition    = [] 
        self.process_field_name() #this initializes the above attributes

        self.time           = None
        self.tsymbol        = None
        self.tunits         = None
        self.Gdata          = []
        self.dims           = None
        self.ndims          = None
        self.cells          = None
        self.grids          = None
        self.values         = None
        self.vol_int        = None
        self.vol_avg        = None

        # attribute to handle slices
        self.dim_idx        = None       
        self.new_grids      = []
        self.new_gnames     = []
        self.new_gsymbols   = []
        self.new_gunits     = []
        self.new_dims       = []
        self.sliceddim      = []
        self.slicecoords    = {}
        self.slicetitle     = ''
        self.timetitle      = ''

        if load:
            self.load(polyorder=polyorder,polytype=polytype)

    def process_field_name(self):
        # Field composition to handle composed fields.
        # Should be just name if its a simple field 
        # but can be, e.g., 'ne','Tperpe' for electron perpendicular pressure.
        # self.composition.append(self.name)
        self.composition = self.simulation.normalization[self.name+'compo']
        # field receipe (function of the gdata)
        self.receipe     = self.simulation.normalization[self.name+'receipe']
        for subname in self.composition:
            subdataname = self.simulation.data_param.data_files_dict[subname+'file']
            self.datanames.append(subdataname)
            if subname in ['b_x','b_y','b_z','Jacobian','Bmag']: #time dependent data, add tf in the name
                name_tf = subdataname
            else: # time independent data (like geom)
                name_tf = '%s_%d'%(subdataname,self.tf)
            self.filenames.append("%s-%s.gkyl"%(self.simulation.data_param.fileprefix,name_tf))
            self.comp.append(self.simulation.data_param.data_files_dict[subname+'comp'])
            self.gnames = copy.deepcopy(self.simulation.data_param.data_files_dict[subname+'gnames'])
        self.gsymbols = []
        self.gunits   = []
        for key in self.gnames:
            self.gsymbols.append(self.simulation.normalization[key+'symbol'])
            self.gunits.append(self.simulation.normalization[key+'units'])
        self.vsymbol = self.simulation.normalization[self.name+'symbol']        
        self.vunits  = self.simulation.normalization[self.name+'units']

    def load(self,polyorder=1,polytype='ms'):
        self.Gdata = []
        for (f_,c_) in zip(self.filenames,self.comp):
            # Load the data from the file
            Gdata = pg.data.GData(f_)
            # Interpolate the data using modal interpolation
            dg = pg.data.GInterpModal(Gdata,polyorder,polytype)
            dg.interpolate(c_,overwrite=True)
            self.Gdata.append(Gdata)
            if Gdata.ctx['time']:
                self.time = Gdata.ctx['time']
        self.grids   = [g for g in Gdata.get_grid() if len(g) > 1]
        self.cells   = Gdata.ctx['cells']
        self.ndims   = len(self.cells)
        self.dim_idx = list(range(self.ndims))
        if not self.time:
            self.time = 0
        self.refresh()
        self.normalize()
        self.rename()

    def refresh(self,values=True):
        self.new_cells    = self.Gdata[0].ctx['cells']
        self.new_grids    = []
        self.new_gnames   = []
        self.new_gsymbols = []
        self.new_gunits   = []
        self.new_dims     = [c_ for c_ in self.new_cells if c_ > 1]
        self.dim_idx      = [d_ for d_ in range(self.ndims) if d_ not in self.sliceddim]
        for idx in self.dim_idx:
            # number of points in the idx grid
            Ngidx = len(self.grids[idx])
            # there is an additional point we have to remove to have similar size with values
            self.new_grids.append(mt.create_uniform_array(self.grids[idx],Ngidx-1))
            # self.new_grids.append(self.grids[idx][:-1])
            self.new_gnames.append(self.gnames[idx])
            self.new_gsymbols.append(self.gsymbols[idx])
            self.new_gunits.append(self.gunits[idx])
        if values:
            # compute again the values
            self.values = copy.deepcopy(self.receipe(self.Gdata))
            self.values = self.values.reshape(self.new_dims)                
        self.values = np.squeeze(self.values)

    def normalize(self,values=True,time=True,grid=True):
        if time:
            # Normalize time
            self.time    /= self.simulation.normalization['tscale']
            self.tsymbol  = self.simulation.normalization['tsymbol']
            self.tunits   = self.simulation.normalization['tunits']
        if grid:
            # Normalize the grids
            for ig in range(len(self.grids)):
                self.grids[ig]   /= self.simulation.normalization[self.gnames[ig]+'scale']
                self.grids[ig]   -= self.simulation.normalization[self.gnames[ig]+'shift']
                self.gsymbols[ig] = self.simulation.normalization[self.gnames[ig]+'symbol']
                self.gunits[ig]   = self.simulation.normalization[self.gnames[ig]+'units']
        if values:
            # Normalize the values
            self.values  /= self.simulation.normalization[self.name+'scale']
            self.values  -= self.simulation.normalization[self.name+'shift']
            self.vsymbol  = self.simulation.normalization[self.name+'symbol']
            self.vunits   = self.simulation.normalization[self.name+'units']

    def rename(self):
        slicetitle = ''
        norm = self.simulation.normalization
        for k_,c_ in self.slicecoords.items():
            if isinstance(c_,float):
                slicetitle += norm[k_+'symbol']+'=%2.2f'%c_ + norm[k_+'units'] +', '
            else:
                slicetitle += c_ +', '

        self.slicetitle = slicetitle
        self.timetitle  = self.tsymbol + '=%2.2f'%self.time+self.tunits
        self.fulltitle  = self.slicetitle + self.timetitle

    def select_slice(self, direction, cut):
        # Map the direction to the corresponding axis index
        direction_map = {'x':0,'y':1,'z':2,'vpar':3,'mu':4}
        
        if direction not in direction_map:
            raise ValueError("Invalid direction '"+direction+"': must be 'x', 'y', 'z', 'vpar', or 'mu'")
        
        ic = direction_map[direction]  # Get the axis index for the given direction

        if cut == 'avg':
            cut_coord   = direction+'-avg'
            self.values = np.average(self.values,axis=ic)
            # grid        = self.grids[ic][:]
            # Jacobian    = self.simulation.geom_param.Jacobian
            # self.values = np.trapz(self.values*Jacobian,grid,axis=ic)
            # self.values/= np.trapz(Jacobian,grid,axis=ic)
        elif cut == 'max':
            cut_coord = direction+'-max'
            self.values = np.max(self.values,axis=ic)
        else:
            # If cut is an integer, use it as the index
            if isinstance(cut, int):
                cut_index = np.minimum(cut,len(self.grids[ic])-2)
            else:
                # Find the closest index in the corresponding grid
                cut_index = (np.abs(self.grids[ic] - cut)).argmin()
                cut_index = np.minimum(cut_index,len(self.grids[ic])-2)
            # find the cut coordinate
            cut_coord = self.grids[ic][cut_index]
            # Select the slice of values at the cut_index along the given direction
            self.values = np.take(np.copy(self.values), cut_index, axis=ic)

        # Expand to avoid dimensionality reduction
        self.values = np.expand_dims(self.values, axis=ic)

        # record the cut and adapt the grids
        self.sliceddim.append(ic)
        self.slicecoords[direction] = cut_coord   
        self.rename()

    def slice_1D(self,cutdirection,ccoords):
        axes = 'xyz'
        axes = axes.replace(cutdirection,'')
        for i_ in range(len(axes)):
            self.select_slice(direction=axes[i_],cut=ccoords[i_])
        self.refresh(values=False)

    def slice_2D(self,plane,ccoord):
        # Select the specific slice dimension indices
        i1 = getgrid_index(plane[0])
        i2 = getgrid_index(plane[1])
        # Define the cut dimension
        i3   = 2*(i1==0 and i2==1)+1*(i1==0 and i2==2)+0*(i1==1 and i2==2)
        sdir = self.gnames[i3]
        # Reduce the dimensionality of the data
        self.select_slice(direction=sdir,cut=ccoord)
        self.refresh(values=False)

    def compute_volume_integral(self,jacob_squared=False):
        # We load the original grid (in original units)
        [x,y,z] = self.simulation.geom_param.grids
         # This is useful for bimax moment that are output divide by jacobian
        if jacob_squared:
            Jac = self.simulation.geom_param.Jacobian**2
        else:
            Jac = self.simulation.geom_param.Jacobian
        self.vol_int = mt.integral_xyz(x,y,z,self.values*Jac)
        return self.vol_int
    
    def compute_volume_average(self,jacob_squared=False):
        # We load the original grid (in original units)
        [x,y,z] = self.simulation.geom_param.grids
        Jac     = self.simulation.geom_param.Jacobian
        # This is useful for bimax moment that are output divide by jacobian
        if jacob_squared:
            Jac *= Jac
        intJac = self.simulation.geom_param.intJac
        self.vol_avg = mt.integral_xyz(x,y,z,self.values*Jac)/intJac
        return self.vol_avg
    
    def free_values(self):
        self.values = None

    def fourrier_y(self):
        # Apply FFT only along the y dimension (axis=1)
        fft_ky = np.fft.rfft(self.values, axis=1)

        # Get the Fourier frequencies for the y-dimension (ky)
        Ny  = self.values.shape[1]  # Number of points in the y-dimension
        y   = self.grids[1]
        dy  = (y[1] - y[0])*self.simulation.normalization['yscale']
        ky  = np.fft.rfftfreq(Ny, d=dy)  # Fourier frequencies for the y-dimension (ky)
        Nky = len(ky)

        # We extend our ky array by one 
        # to follow the N+1 format of the grids
        ky = mt.create_uniform_array(ky,Nky+1)

        # Remove the zeroth wavenumber
        ky     = ky[1:]
        fft_ky = fft_ky[:,1:,:]

        # Update frame data
        self.values   = np.abs(fft_ky)
        # renaming the y axis (wavelength)
        # gname = 'wavelen'
        # self.grids[1]    = 1.0/ky
        # self.gnames[1]   = gname
        # self.gsymbols[1] = self.simulation.normalization[gname+'symbol']
        # self.gunits[1]   = self.simulation.normalization[gname+'units']

        # renaming the y axis (wavenumber)
        gname = 'ky'
        self.grids[1]    = ky
        self.gnames[1]   = gname
        self.gsymbols[1] = self.simulation.normalization[gname+'symbol']
        self.gunits[1]   = self.simulation.normalization[gname+'units']

        # refresh everything
        self.refresh(values=False)