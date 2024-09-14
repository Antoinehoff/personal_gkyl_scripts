import sys
import postgkyl as pg


def getgrid_index(s):
    return  (1*(s == 'x') + 2*(s == 'y') + 3*(s == 'z') + 4*(s == 'vpar') + 5*(s == 'mu'))-1

class Frame:
    def __init__(self,simulation,name,tf):
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
        self.Gdata          = None
        self.dims           = None
        self.ndims          = None
        self.cells          = None
        self.grids          = None
        self.values         = None
        self.slicecoords    = []
        self.slicetitle     = None

    def load(self,polytype='ms'):
        # Load the data from the file
        Gdata = pg.data.GData(self.filename)
        # Interpolate the data using modal interpolation
        dg = pg.data.GInterpModal(Gdata,1,polytype)
        dg.interpolate(self.comp,overwrite=True)
        self.Gdata = Gdata
        self.grids  = [g for g in self.Gdata.get_grid() if len(g) > 1]
        self.time  = Gdata.ctx['time']
        self.refresh()
        self.normalize()

    def refresh(self):
        self.cells  = self.Gdata.ctx['cells']
        new_gnames   = []
        new_gsymbols = []
        new_gunits   = []
        new_dims     = [c_ for c_ in self.cells if c_ > 1]
        for i in range(len(new_dims)):
            new_gnames.append(self.gnames[i])
            new_gsymbols.append(self.gsymbols[i])
            new_gunits.append(self.gunits[i])
        self.dims     = new_dims
        self.gnames   = new_gnames
        self.gsymbols = new_gsymbols
        self.gunits   = new_gunits
        self.ndims    = len(self.dims)
        self.values   = self.Gdata.get_values()
        self.values   = self.values.reshape(self.dims)

    def normalize(self):
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

    def slice_1D(self,cutdirection,ccoords):
        # Select the specific slice
        icut = getgrid_index(cutdirection)
        i0 = icut
        i1 = (icut+1)%self.ndims
        i2 = (icut+2)%self.ndims
        self.grids  = (self.grids[i0][:-1]+self.grids[i0][1:])/2
        if i0 == 0:
            pg.data.select(self.Gdata, z1=ccoords[0], z2=ccoords[1], overwrite=True)
        elif i0 == 1:
            pg.data.select(self.Gdata, z0=ccoords[0], z2=ccoords[1], overwrite=True)
        elif i0 == 2:
            pg.data.select(self.Gdata, z0=ccoords[0], z1=ccoords[1], overwrite=True)
        c1 = (self.Gdata.ctx['lower'][i1]+self.Gdata.ctx['upper'][i1])/2.
        c2 = (self.Gdata.ctx['lower'][i2]+self.Gdata.ctx['upper'][i2])/2.
        self.slicetitle = self.gsymbols[i1]+'=%2.2f'%c1+self.gunits[i1] \
            +', '+ self.gsymbols[i2]+'=%2.2f'%c2+self.gunits[i2]
        self.gsymbols = [self.gsymbols[i0]]
        self.gunits   = [self.gunits[i0]]
        self.slicecoords = [c1,c2]
        self.refresh()

    def compress(self,direction,type='cut',cut=0):
        if direction == 'x':
            pg.data.select(self.Gdata, z0=cut, overwrite=True)
        elif direction == 'y':
            pg.data.select(self.Gdata, z1=cut, overwrite=True)
        elif direction == 'z':
            pg.data.select(self.Gdata, z2=cut, overwrite=True)
        self.refresh()

    def slice_2D(self,plane,ccoord):
        # Select the specific slice
        i1 = getgrid_index(plane[0])
        i2 = getgrid_index(plane[1])
        if plane == ['x','y']:
            self.gsymbols = [self.gsymbols[i1],self.gsymbols[i2]]
            self.gunits   = [self.gunits[i1],  self.gunits[i2]]
            x1  = (self.grids[i1][:-1]+self.grids[i1][1:])/2
            x2  = (self.grids[i2][:-1]+self.grids[i2][1:])/2
            pg.data.select(self.Gdata, z2=ccoord, overwrite=True)
            cc = (self.Gdata.ctx['lower'][2]+self.Gdata.ctx['upper'][2])/2.
            self.slicetitle = "$z=%2.2f$"%(cc)
        elif plane == ['x','z']:
            self.gsymbols = [self.gsymbols[0],self.gsymbols[2]]
            self.grids  = (self.grids[1][:-1]+self.grids[1][1:])/2
            pg.data.select(self.Gdata, z0=ccoords[0], z2=ccoords[1], overwrite=True) 
            c1 = (self.Gdata.ctx['lower'][0]+self.Gdata.ctx['upper'][0])/2.
            c2 = (self.Gdata.ctx['lower'][2]+self.Gdata.ctx['upper'][2])/2.     
            self.slicetitle = "$x=%2.2f$ (m), $z=%2.2f$"%(c1,c2)
        elif plane == ['y','z']:
            self.gsymbols = [self.gsymbols[1],self.gsymbols[2]]
            self.grids  = (self.grids[2][:-1]+self.grids[2][1:])/2
            pg.data.select(self.Gdata, z0=ccoords[0], z1=ccoords[1], overwrite=True)
            c1 = (self.Gdata.ctx['lower'][0]+self.Gdata.ctx['upper'][0])/2.
            c2 = (self.Gdata.ctx['lower'][1]+self.Gdata.ctx['upper'][1])/2.
            self.slicetitle = "$x=%2.2f$ (m), $y=%2.2f$ (m)"%(c1,c2)
        self.grids = [x1, x2]
        self.slicecoords = [cc]
        self.refresh()

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