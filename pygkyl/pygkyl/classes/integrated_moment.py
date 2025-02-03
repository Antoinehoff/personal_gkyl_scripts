from ..tools import pgkyl_interface as pgkyl_
import numpy as np

class Integrated_moment:
    simulation = None
    name = None
    spec_s = None
    values = None
    receipe = None
    scale = None
    vunits = None
    tunits = None
    symbol = None
    time = None

    def __init__(self, simulation, name, load=True, ddt=False):
        self.simulation = simulation
        self.name = name
        if name[-1] == 'e': self.spec_s = 'elc'
        elif name[-1] == 'i': self.spec_s = 'ion'
        elif name[-3:] == 'tot': self.spec_s = ['elc','ion']

        self.set_units_and_labels()

        if load: self.load()
        if ddt: self.ddt()

    def set_units_and_labels(self):
        simulation = self.simulation
        spec_s = self.spec_s
        if self.name[:-1] in ['n']:
            def receipe(x): return x[:,0]
            scale = 1.0
            vunits = 'particles'
            symbol = r'$\bar n_%s$'%spec_s[0]
        elif self.name[:-1] in ['upar']:
            def receipe(x): return x[:,1]
            scale = simulation.species[spec_s].m*simulation.species[spec_s].vt
            vunits = ''
            symbol = r'$\bar u_{\parallel %s}/v_{t %s}$'%(spec_s[0],spec_s[0])
        elif self.name[:-1] in ['Tpar']:
            def receipe(x): return x[:,2]
            scale = simulation.species[spec_s].m
            vunits = 'eV'
            symbol = r'$\bar T_{\parallel %s}$'%spec_s[0]
        elif self.name[:-1] in ['Tperp']:
            def receipe(x): return x[:,3]
            scale = simulation.species[spec_s].m
            vunits = 'eV'
            symbol = r'$\bar T_{\perp %s}$'%spec_s[0]
        elif self.name[:-1] in ['T']:
            def receipe(x): return 1/3*(x[:,2]+2*x[:,3])
            scale = simulation.species[spec_s].m
            vunits = 'eV'
            symbol = r'$\bar T_{%s}$'%spec_s[0]
        elif self.name[:-1] in ['W','Pow']:
            def receipe(x): return 1/3*(x[:,2]+2*x[:,3])
            scale = simulation.species[spec_s].m / 1e6                
            vunits = 'MJ'
            symbol = r'$W_{kin,%s}$'%spec_s[0]
        elif self.name in ['Wtot']:
            def receipe(x): return 1/3*(x[:,2]+2*x[:,3])
            scale = [simulation.species[s].m / 1e6 for s in spec_s]
            vunits = 'MJ'
            symbol = r'$W_{kin,tot}$'
        elif self.name in ['ntot']:
            def receipe(x): return x[:,0]
            scale = [1.0 for s in spec_s]      
            vunits = 'particles'
            symbol = r'$\bar n_{tot}$'    

        self.receipe = receipe
        self.scale = scale
        self.vunits = vunits
        self.tunits = self.simulation.normalization.dict['tunits']
        self.symbol = symbol    

    def load(self):
        """
        Load the integrated moment data from the simulation.
        """
        self.values = 0
        if isinstance(self.spec_s,str):
            f_ = self.simulation.data_param.fileprefix+'-'+self.spec_s+'_integrated_moms.gkyl'
            Gdata = pgkyl_.get_gkyl_data(f_)
            self.values = pgkyl_.get_values(Gdata) * self.scale
        else:
            for s in self.spec_s:
                f_ = self.simulation.data_param.fileprefix+'-'+s+'_integrated_moms.gkyl'
                Gdata = pgkyl_.get_gkyl_data(f_)
                self.values += pgkyl_.get_values(Gdata) * self.scale[self.spec_s.index(s)]

        self.time = np.squeeze(Gdata.get_grid()) / self.simulation.normalization.dict['tscale']
        self.values = self.receipe(self.values)
        self.values = np.squeeze(self.values)
        # remove double diagnostic
        self.time, indices = np.unique(self.time, return_index=True)
        self.values = self.values[indices]

    def ddt(self):
        """
        Calculate the time derivative of the integrated moment.
        """
        self.values = np.gradient(self.values, self.time)
        if self.vunits[-1] == 'J':
            self.vunits.replace('J','W')

        else:
            self.vunits = self.vunits + '/s'
        self.symbol = r'$\partial$'+self.symbol+r'/$\partial t$'