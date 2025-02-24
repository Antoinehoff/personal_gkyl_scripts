from ..tools import pgkyl_interface as pgkyl_
import numpy as np

class IntegratedMoment:
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
    momtype = None

    def __init__(self, simulation, name, load=True, ddt=False):
        self.simulation = simulation
        self.name = name
        if name[-1] == 'e': self.spec_s = 'elc'
        elif name[-1] == 'i': self.spec_s = 'ion'
        elif name[-3:] == 'tot': self.spec_s = ['elc','ion']

        self.detect_momtype()

        self.set_units_and_labels()

        if load: self.load()
        if ddt: self.ddt()

    def set_units_and_labels(self):
        simulation = self.simulation
        spec_s = self.spec_s
        if self.momtype in ['BiMaxwellian','bimaxwellian']:
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
            else:
                print(self.name + ' is not available as integrated moment.')
                print('The available fields are: ns, upars, Tpars, tperps, Ts, Ws, Wtot, ntot. (for s=i,e).')
                raise ValueError('Provided fieldname is not available')
        elif self.momtype in ['Hamiltonian','hamiltonian']:
            if self.name[:-1] in ['n']:
                def receipe(x): return x[:,0]
                scale = 1.0
                vunits = 'particles'
                symbol = r'$\bar n_%s$'%spec_s[0]
            elif self.name[:-1] in ['upar']:
                def receipe(x): return x[:,1]
                scale = simulation.species[spec_s].vt
                vunits = ''
                symbol = r'$\bar u_{\parallel %s}/v_{t %s}$'%(spec_s[0],spec_s[0])
            elif self.name[:-1] in ['W','H']:
                def receipe(x): return x[:,2]
                scale = 1.0 / 1e6                
                vunits = 'MJ'
                symbol = r'$H_{%s}$'%spec_s[0]
            elif self.name in ['Wtot','Htot']:
                def receipe(x): return x[:,2]
                scale = [1.0 / 1e6 for s in spec_s]
                vunits = 'MJ'
                symbol = r'$H_{tot}$'
            elif self.name in ['ntot']:
                def receipe(x): return x[:,0]
                scale = [1.0 for s in spec_s]      
                vunits = 'particles'
                symbol = r'$\bar n_{tot}$'    
            else:
                print(self.name + ' is not available as integrated moment.')
                print('The available fields are: ns, upars, Tpars, tperps, Ts, Ws, Wtot, ntot. (for s=i,e).')
                raise ValueError('Provided fieldname is not available')
        else:
            print(self.momtype+' not available. Must be BiMaxwellian or Hamiltonian')
            raise ValueError('Provided momtype is not available')
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
        # detect if we are in Hamiltonian or BiMaxwellian diagnostic by getting the number of components
        self.momtype = 'BiMaxwellian' if Gdata.get_num_comps == 4 else 'Hamiltonian'

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

    def detect_momtype(self):
        """
        Check if we are analyzing Hamiltonian or Bimaxwellian moments (3 vs 4 components)
        """
        if isinstance(self.spec_s,str):
            species = self.spec_s
        else:
            species = self.spec_s[0]
        f_ = self.simulation.data_param.fileprefix+'-'+species+'_integrated_moms.gkyl'
        Gdata = pgkyl_.get_gkyl_data(f_)
        self.momtype = 'BiMaxwellian' if Gdata.get_num_comps == 4 else 'Hamiltonian'