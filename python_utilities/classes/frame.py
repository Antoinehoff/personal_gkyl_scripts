import sys
import postgkyl as pg


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
        self.dataname       = []
        self.filename       = []
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
        if self.name[0] == 'p' and self.name[1] != 'h': #detect if we are dealing with pressure
            self.composition.append('n%s'%self.name[-1]) # add density to the composition
            self.composition.append('T%s'%self.name[1:]) # add corresponding temperature
        else:
            self.composition.append(self.name)
        for subname in self.composition:
            subdataname = self.simulation.data_param.data_files_dict[subname+'file']
            self.dataname.append(subdataname)
            self.filename.append("%s-%s_%d.gkyl"%(
                self.simulation.data_param.fileprefix,subdataname,self.tf))
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
        for (f_,c_) in zip(self.filename,self.comp):
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
        self.refresh()
        self.normalize()
        self.rename()

    def refresh(self):
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
        # compute again the values
        self.compute_field()

    def compute_field(self):
        self.values = 1.0
        for gdata_ in self.Gdata:
            self.values *= gdata_.get_values()
        self.values   = self.values.reshape(self.new_dims)

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

    def slice_1D(self,cutdirection,ccoords):
        axes = 'xyz'
        axes = axes.replace(cutdirection,'')
        for i_ in range(len(axes)):
            self.compress(direction=axes[i_],type='cut',cut=ccoords[i_])


    def slice_2D(self,plane,ccoord):
        # Select the specific slice dimension indices
        i1 = getgrid_index(plane[0])
        i2 = getgrid_index(plane[1])
        # Define the cut dimension
        i3   = 2*(i1==0 and i2==1)+1*(i1==0 and i2==2)+0*(i1==1 and i2==2)
        sdir = self.gnames[i3]
        # Reduce the dimensionality of the data
        self.compress(sdir,type='cut',cut=ccoord)

    def free_values(self):
        self.values = None