import sys
import postgkyl as pg


def getgrid_index(s):
    return  (1*(s == 'x') + 2*(s == 'y') + 3*(s == 'z') + 4*(s == 'v') + 5*(s == 'm'))-1

class Frame:
    def __init__(self,simulation,name,tf,load=False,polytype='ms'):
        """
        Initialize a Frame instance with all attributes set to None.
        """
        self.simulation     = simulation
        self.name           = name
        self.tf             = tf
        self.dataname       = None
        self.filename       = None
        self.comp           = None
        self.gnames         = None
        self.gsymbols       = None
        self.gunits         = None
        self.vsymbol        = None
        self.vunits         = None
        self.process_field_name() #this initializes the above attributes
        self.time           = None
        self.tsymbol        = None
        self.tunits         = None
        self.Gdata          = None
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
            self.load(polytype=polytype)

    def load(self,polytype='ms'):
        # Load the data from the file
        Gdata = pg.data.GData(self.filename)
        # Interpolate the data using modal interpolation
        dg = pg.data.GInterpModal(Gdata,1,polytype)
        dg.interpolate(self.comp,overwrite=True)
        # Divide by density if we look for fluid velocity
        if self.name[0] == 'u':
            # we should divide here u/n when not BiMaxwellian
            u=1
        self.Gdata   = Gdata
        self.grids   = [g for g in self.Gdata.get_grid() if len(g) > 1]
        self.cells   = self.Gdata.ctx['cells']
        self.ndims   = len(self.cells)
        self.dim_idx = list(range(self.ndims))
        self.time    = Gdata.ctx['time']
        self.refresh()
        self.normalize()

    def refresh(self):
        self.new_cells    = self.Gdata.ctx['cells']
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
        self.values   = self.Gdata.get_values()
        self.values   = self.values.reshape(self.new_dims)

    def normalize(self):
        # Normalize time
        self.time    /= self.simulation.normalization['tscale']
        self.tsymbol  = self.simulation.normalization['tsymbol']
        self.tunits   = self.simulation.normalization['tunits']
        # Normalize the grids
        for ig in range(len(self.grids)):
            self.grids[ig]   /= self.simulation.normalization[self.gnames[ig]+'scale']
            self.grids[ig]   -= self.simulation.normalization[self.gnames[ig]+'shift']
            self.gsymbols[ig] = self.simulation.normalization[self.gnames[ig]+'symbol']
            self.gunits[ig]   = self.simulation.normalization[self.gnames[ig]+'units']
        # Normalize the values
        self.values  /= self.simulation.normalization[self.name+'scale']
        self.values  -= self.simulation.normalization[self.name+'shift']
        self.vsymbol  = self.simulation.normalization[self.name+'symbol']
        self.vunits   = self.simulation.normalization[self.name+'units']
        slicetitle = ''
        norm = self.simulation.normalization
        for k_,c_ in self.slicecoords.items():
            slicetitle += norm[k_+'symbol']+'=%2.2f, '%c_ + norm[k_+'units']
        self.slicetitle = slicetitle
        self.timetitle  = self.tsymbol + '=%2.2f'%self.time+self.tunits

    def compress(self,direction,type='cut',cut=0):
        if direction == 'x':
            ic = 0
            pg.data.select(self.Gdata, z0=cut, overwrite=True)
        elif direction == 'y':
            ic = 1
            pg.data.select(self.Gdata, z1=cut, overwrite=True)
        elif direction == 'z':
            ic = 2
            pg.data.select(self.Gdata, z2=cut, overwrite=True)
        elif direction == 'v':
            ic = 3
            pg.data.select(self.Gdata, z3=cut, overwrite=True)
        elif direction == 'm':
            ic = 4
            pg.data.select(self.Gdata, z4=cut, overwrite=True)
        self.sliceddim.append(ic)
        cc = (self.Gdata.ctx['lower'][ic]+self.Gdata.ctx['upper'][ic])/2.
        self.slicecoords[direction] = cc   
        self.refresh()

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

    def process_field_name(self):
        self.dataname = self.simulation.data_param.data_files_dict[self.name+'file']
        self.filename = "%s-%s_%d.gkyl"%(
            self.simulation.data_param.fileprefix,self.dataname,self.tf)
        self.comp     = self.simulation.data_param.data_files_dict[self.name+'comp']
        self.gnames   = self.simulation.data_param.data_files_dict[self.name+'gnames']
        self.gsymbols = []
        self.gunits   = []
        for key in self.gnames:
            self.gsymbols.append(self.simulation.normalization[key+'symbol'])
            self.gunits.append(self.simulation.normalization[key+'units'])
        self.vsymbol = self.simulation.normalization[self.name+'symbol']        
        self.vunits  = self.simulation.normalization[self.name+'units']

    def free_values(self):
        self.values = None