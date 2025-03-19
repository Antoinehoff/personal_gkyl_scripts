from ..tools import pgkyl_interface as pgkyl_
import numpy as np

class IntegratedMoment:
    """
    Class to load and analyze integrated moments from a simulation.
    
    Possible integrated moments are:
    - ns: number density
    - upars: parallel velocity
    - Tpars: parallel temperature
    - Tperps: perpendicular temperature
    - Ts: total temperature
    - Ws: kinetic energy
    - Wtot: total kinetic energy
    - ntot: total number density
    - Hs: Hamiltonian
    - Htot: total Hamiltonian
    - you can add 'src_' in front of the name to get the source term of the integrated moment
    - you can add 'bflux_d_e_' to get the boundary flux d=x,y,z direction at the e=u,l edge
    """
    simulation = None
    momname = None
    spec_s = None
    values = None
    receipe = None
    scale = None
    vunits = None
    tunits = None
    symbol = None
    time = None
    momtype = None
    bflux = False
    direction = None
    edge = None
    fluxextension = ''
    fluxname = ''
    bflux_list = ''
    issource = False
    src = ''

    def __init__(self, simulation, name, load=True, ddt=False):
        
        self.simulation = simulation

        self.process_name(name)
        self.detect_momtype()
        self.set_units_and_labels()

        if load: self.load()
        if ddt: self.ddt()

    def set_units_and_labels(self):
        simulation = self.simulation
        spec_s = self.spec_s
        if self.momtype == 'BiMaxwellian':
            if self.momname[:-1] in ['n']:
                def receipe(x): return x[:,0]
                scale = 1.0
                vunits = 'particles'
                symbol = r'$\bar n_%s$'%spec_s[0]
            elif self.momname[:-1] in ['upar']:
                def receipe(x): return x[:,1]
                scale = simulation.species[spec_s].m*simulation.species[spec_s].vt
                vunits = ''
                symbol = r'$\bar u_{\parallel %s}/v_{t %s}$'%(spec_s[0],spec_s[0])
            elif self.momname[:-1] in ['Tpar']:
                def receipe(x): return x[:,2]
                scale = simulation.species[spec_s].m
                vunits = 'eV'
                symbol = r'$\bar T_{\parallel %s}$'%spec_s[0]
            elif self.momname[:-1] in ['Tperp']:
                def receipe(x): return x[:,3]
                scale = simulation.species[spec_s].m
                vunits = 'eV'
                symbol = r'$\bar T_{\perp %s}$'%spec_s[0]
            elif self.momname[:-1] in ['T']:
                def receipe(x): return 1/3*(x[:,2]+2*x[:,3])
                scale = simulation.species[spec_s].m
                vunits = 'eV'
                symbol = r'$\bar T_{%s}$'%spec_s[0]
            elif self.momname[:-1] in ['W','Pow']:
                def receipe(x): return 1/3*(x[:,2]+2*x[:,3])
                scale = simulation.species[spec_s].m / 1e6                
                vunits = 'MJ'
                symbol = r'$W_{kin,%s}$'%spec_s[0]
            elif self.momname in ['Wtot']:
                def receipe(x): return 1/3*(x[:,2]+2*x[:,3])
                scale = [simulation.species[s].m / 1e6 for s in spec_s]
                vunits = 'MJ'
                symbol = r'$W_{kin,tot}$'
            elif self.momname in ['ntot']:
                def receipe(x): return x[:,0]
                scale = [1.0 for s in spec_s]      
                vunits = 'particles'
                symbol = r'$\bar n_{tot}$'  
            else:
                print(self.momname + ' is not available as integrated moment.')
                print('The available fields are: ns, upars, Tpars, tperps, Ts, Ws, Wtot, ntot. (for s=i,e).')
                raise ValueError('Provided fieldname is not available')
        elif self.momtype == 'Hamiltonian':
            if self.momname[:-1] in ['n']:
                def receipe(x): return x[:,0]
                scale = 1.0
                vunits = 'particles'
                symbol = r'$\bar n_%s$'%spec_s[0]
            elif self.momname[:-1] in ['upar']:
                def receipe(x): return x[:,1]
                scale = simulation.species[spec_s].vt
                vunits = ''
                symbol = r'$\bar u_{\parallel %s}/v_{t %s}$'%(spec_s[0],spec_s[0])
            elif self.momname[:-1] in ['W','H']:
                def receipe(x): return x[:,2]
                scale = 1.0 / 1e6                
                vunits = 'MJ'
                symbol = r'$H_{%s}$'%spec_s[0]
            elif self.momname in ['Wtot','Htot']:
                def receipe(x): return x[:,2]
                scale = [1.0 / 1e6 for s in spec_s]
                vunits = 'MJ'
                symbol = r'$H_{tot}$'
            elif self.momname in ['ntot']:
                def receipe(x): return x[:,0]
                scale = [1.0 for s in spec_s]      
                vunits = 'particles'
                symbol = r'$\bar n_{tot}$'    
            else:
                print(self.momname + ' is not available as integrated moment.')
                print('The available fields are: ns, upars, Hs, Htot, ntot. (for s=i,e).')
                raise ValueError('Provided fieldname is not available')
        self.receipe = receipe
        self.scale = scale if isinstance(scale,list) else [scale]
        self.vunits = vunits
        self.tunits = self.simulation.normalization.dict['tunits']
        self.symbol = symbol
        if self.bflux:
            self.symbol = self.fluxname+r'('+self.symbol+r')'
            self.vunits += '/s' if self.vunits else '1/s'
        if self.issource:
            self.symbol = r'S('+self.symbol+r')'
            self.vunits += '/s'

    def load(self):
        """
        Load the integrated moment data from the simulation.
        """
        self.values = 0
        species_list = self.spec_s if isinstance(self.spec_s,list) else [self.spec_s]
        for s_ in species_list:
            for bf_ in self.bflux_list:
                f_ = self.simulation.data_param.fileprefix+'-'+s_+self.src+bf_+'_integrated_'
                if not pgkyl_.file_exists(f_+'moms.gkyl'):
                    f_ += self.momtype+'Moments.gkyl'
                    if not pgkyl_.file_exists(f_):
                        raise FileNotFoundError(f_ + ' does not exist.')
                else:
                    f_ += 'moms.gkyl'
                Gdata = pgkyl_.get_gkyl_data(f_)
                self.values += pgkyl_.get_values(Gdata) * self.scale[species_list.index(s_)]

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
        self.values = np.gradient(self.values, self.time * self.simulation.normalization.dict['tscale'], edge_order=2)
        if 'J' in self.vunits:
            self.vunits = self.vunits.replace('J','W')
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
        datashape = np.shape(pgkyl_.get_values(Gdata))
        self.momtype = 'BiMaxwellian' if datashape[1] == 4 else 'Hamiltonian'
        
    def process_name(self,name):
        # convention: write bflux_x_u_fieldname with x the direction (x or z) and u the edge (u or l)
        # or bflux_alldir
        nres = '' # residual name after removing boundary flux extensions
        if 'bflux' in name:
            self.bflux = True
            nres = name
            nres = nres.replace('bflux_','')
            self.fluxname = r'$F$'
            if nres[0] in ['x','z']:
                self.direction = nres[0]
                nres = nres.replace(self.direction+'_','')
                self.fluxname += r'$_x$' if self.direction == 'x' else r'$_\parallel$'
                if nres[0] in ['u','l']:
                    self.edge = nres[0]
                    nres = nres.replace(self.edge+'_','')
                    self.fluxname += r'$^{up}$' if self.edge == 'u' else r'$^{lo}$'
                elif nres[0:5] == 'total':
                    self.edge = ['u','l']
                    nres = nres.replace('total_','')
            elif nres[0:5] == 'total':
                self.direction = ['x','z']
                self.edge = ['u','l']
                nres = nres.replace('total_','')

            self.bflux_list = []
            dl = self.direction if isinstance(self.direction,list) else [self.direction] 
            el = self.edge if isinstance(self.edge,list) else [self.edge] 
            for d_ in dl:
                for e_ in el:
                    edgestring = 'lower' if e_=='l' else 'upper'
                    self.bflux_list.append('_'+'bflux_'+d_+edgestring)
                    
        else: self.bflux_list = ['']
        self.momname = nres if nres else name
        
        if 'src_' in self.momname:
            self.src = '_source'
            self.momname = self.momname.replace('src_','')
            self.issource = True
        if name[-1] == 'e': self.spec_s = 'elc'
        elif name[-1] == 'i': self.spec_s = 'ion'
        elif name[-3:] == 'tot': self.spec_s = ['elc','ion']