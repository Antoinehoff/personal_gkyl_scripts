import sys
import postgkyl as pg
import numpy as np

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
            self.gnames   = self.simulation.data_param.data_files_dict[subname+'gnames']
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
        self.grids   = [g for g in Gdata.get_grid() if len(g) > 1]
        self.cells   = Gdata.ctx['cells']
        self.ndims   = len(self.cells)
        self.dim_idx = list(range(self.ndims))
        self.time    = Gdata.ctx['time']
        if self.time == None:
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
            self.new_grids.append(self.grids[idx][:-1])
            self.new_gnames.append(self.gnames[idx])
            self.new_gsymbols.append(self.gsymbols[idx])
            self.new_gunits.append(self.gunits[idx])
        if values:
            # compute again the values
            self.values = self.receipe(self.Gdata)
        self.values = self.values.reshape(self.new_dims)        

    def refresh_new(self):
        self.new_grids    = []
        self.new_gnames   = []
        self.new_gsymbols = []
        self.new_gunits   = []
        self.new_dims     = [c_ for c_ in np.shape(self.values) if c_ > 1]
        self.dim_idx      = [d_ for d_ in range(self.ndims) if d_ not in self.sliceddim]
        for idx in self.dim_idx:
            self.new_grids.append(self.grids[idx][:-1])
            self.new_gnames.append(self.gnames[idx])
            self.new_gsymbols.append(self.gsymbols[idx])
            self.new_gunits.append(self.gunits[idx])
        self.values = self.values.reshape(self.new_dims)        


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
            slicetitle += norm[k_+'symbol']+'=%2.2f'%c_ + norm[k_+'units'] +', '
        self.slicetitle = slicetitle
        self.timetitle  = self.tsymbol + '=%2.2f'%self.time+self.tunits
        self.fulltitle  = self.slicetitle + self.timetitle

    def compress(self,direction,type='cut',cut=0):
        ### This function does not work with phi derivatives since
        #   the derivative is meant to be computed after the slice 
        for gdata_ in self.Gdata:
            if direction == 'x':
                ic = 0
                pg.data.select(gdata_, z0=cut, overwrite=True)
            elif direction == 'y':
                ic = 1
                pg.data.select(gdata_, z1=cut, overwrite=True)
            elif direction == 'z':
                ic = 2
                pg.data.select(gdata_, z2=cut, overwrite=True)
            elif direction == 'v':
                ic = 3
                pg.data.select(gdata_, z3=cut, overwrite=True)
            elif direction == 'm':
                ic = 4
                pg.data.select(gdata_, z4=cut, overwrite=True)
        self.sliceddim.append(ic)
        cc = (self.Gdata[0].ctx['lower'][ic]+self.Gdata[0].ctx['upper'][ic])/2.
        self.slicecoords[direction] = cc   
        self.refresh()
        self.normalize(values=True,time=False,grid=False)
        self.rename()

    def select_slice(self, direction, cut):
        # Determine the direction (x, y, z correspond to 0, 1, 2)
        if direction == 'x':
            ic = 0
        elif direction == 'y':
            ic = 1
        elif direction == 'z':
            ic = 2
        else:
            raise ValueError("Invalid direction: must be 'x', 'y', or 'z'")

        # If cut is an integer, use it as the index
        if isinstance(cut, int):
            cut_index = np.minimum(cut,len(self.grids[ic]))
        else:
            # Find the closest index in the corresponding grid
            cut_index = (np.abs(self.grids[ic] - cut)).argmin()
        # find the cut coordinate
        cut_coord = self.grids[ic][cut_index]

        # Select the slice of values at the cut_index along the given direction
        # For direction 'x' (ic=0), we're slicing along the first axis (axis=0)
        if direction == 'x':
            self.values = self.values[cut_index:cut_index+1, :, :]  # Slice along the first axis
        elif direction == 'y':
            self.values = self.values[:, cut_index:cut_index+1, :]  # Slice along the second axis
        elif direction == 'z':
            self.values = self.values[:, :, cut_index:cut_index+1]  # Slice along the third axis

        # record the cut and adapt the grids
        self.sliceddim.append(ic)
        self.slicecoords[direction] = cut_coord   
        self.rename()

    def slice_1D(self,cutdirection,ccoords):
        axes = 'xyz'
        axes = axes.replace(cutdirection,'')
        for i_ in range(len(axes)):
            # self.compress(direction=axes[i_],type='cut',cut=ccoords[i_])
            self.select_slice(direction=axes[i_],cut=ccoords[i_])
        self.refresh_new()

    def slice_2D(self,plane,ccoord):
        # Select the specific slice dimension indices
        i1 = getgrid_index(plane[0])
        i2 = getgrid_index(plane[1])
        # Define the cut dimension
        i3   = 2*(i1==0 and i2==1)+1*(i1==0 and i2==2)+0*(i1==1 and i2==2)
        sdir = self.gnames[i3]
        # Reduce the dimensionality of the data
        # self.compress(sdir,type='cut',cut=ccoord)
        self.select_slice(direction=sdir,cut=ccoord)
        self.refresh_new()

    def free_values(self):
        self.values = None