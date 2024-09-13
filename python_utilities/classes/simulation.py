import numpy as np
import postgkyl as pg
from .numparam import NumParam
from .physparam import PhysParam
from .dataparam import DataParam
from .species   import Species
from .geomparam import GeomParam
class Simulation:
    def __init__(self):
        # Initialize all attributes to None
        self.phys_param = None
        self.num_param  = None
        self.data_param = None
        self.geom_param = None
        self.species    = {}
        self.normalization = {}
        self.init_normalization()

    def set_phys_param(self, eps0, eV, mp, me, B_axis):
        self.phys_param = PhysParam(eps0, eV, mp, me, B_axis)
    
    def set_geom_param(self, R_axis, Z_axis, R_LCFSmid, a_shift, q0, kappa, delta):
        self.geom_param = GeomParam(R_axis, Z_axis, R_LCFSmid, a_shift, q0, kappa, delta)

    def set_data_param(self, expdatadir, g0simdir, simname, simdir, fileprefix, wkdir):
        self.data_param = DataParam(expdatadir, g0simdir, simname, simdir, fileprefix, wkdir)
        self.set_num_param()

    def set_num_param(self):
        data = pg.data.GData(self.data_param.datadir+'xyz.gkyl')
        normgrids = data.get_grid()
        normx = normgrids[0]; normy = normgrids[1]; normz = normgrids[2]
        Nx = (normx.shape[0]-2)*2
        Ny = (normy.shape[0]-2)*2
        Nz = normz.shape[0]*4
        self.num_param = NumParam(Nx, Ny, Nz, Nvp=None, Nmu=None)  # Initialize the grid instance

    def set_species(self,name, m, q, T0, n0):
        s_ = Species(name, m, q, T0, n0)
        s_.set_gyromotion(self.phys_param.B_axis)
        self.species[name] = s_
    
    def set_normalization(self,key,scale,shift,symbol,units):
        self.normalization[key+'scale']  = scale
        self.normalization[key+'shift']  = shift
        self.normalization[key+'symbol'] = symbol
        self.normalization[key+'units']  = units

    def init_normalization(self):
        keys = [
            'x','y','z','vpar','mu','phi',
            'ne','ni','upare','upari',
            'Tpare','Tpari','Tperpe','Tperpi'
            ]
        defaultsymbols = [
            r'$x$',r'$y$',r'$z$',r'$v_\parallel$',r'$\mu$',r'$\phi$',
            r'$n_e$',r'$n_i$',r'$u_{\parallel e}$',r'$u_{\parallel i}$',
            r'$T_{\parallel e}$',r'$T_{\parallel i}$',r'$T_{\perp e}$',r'$T_{\perp i}$'
            ]
        defaultunits = [
            '(m)', '(m)', '', '(m/s)', '(J/T)', '(V)',
            r'(m$^{-3}$)', r'(m$^{-3}$)', '(m/s)', '(m/s)',
            '(J/kg)', '(J/kg)', '(J/kg)', 'J/kg' 
        ]
        symbols = {keys[i]: defaultsymbols[i] for i in range(len(keys))}
        units   = {keys[i]: defaultunits[i]   for i in range(len(keys))}

        [self.set_normalization(key,1,0,symbols[key],units[key]) for key in keys]
