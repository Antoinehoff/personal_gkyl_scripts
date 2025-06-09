import numpy as np
from .dataparam import DataParam

class Normalization:
    """
    Manages the normalization of simulation data.

        Methods:
    - __init__: Initializes the Normalization object.
    - reset: Resets the normalization of a key to the default value.
    - change: change the normalization of a key.
    - set: Normalizes a key based on the specified normalization type.
    - help: Displays available normalizations.
    - info: Displays the normalization log.
    """

    def __init__(self,simulation):
        self.simulation = simulation
        self.dict = DataParam.get_default_units_dict(simulation.species)
        self.norm_log = []

    def reset(self,key = None):
        # Get the default dictionary
        default_dict = DataParam.get_default_units_dict(self.simulation.species)
        if key is None:
            # reset all keys to default values
            for k in self.dict.keys():
                if k in default_dict.keys():
                    self.dict[k] = default_dict[k]
            return
        if key not in default_dict.keys():
            print(f"Warning: The key '{key}' is not recognized. No reset performed.")
            return
        # Reset the normalization of key to the default value
        adds = ['scale','shift','symbol','units']
        for add in adds:
            self.dict[key+add]  = default_dict[key+add]

    def change(self,key,scale,shift,symbol,units):
        """
        Set the normalization parameters for a given field.

        Parameters:
        key (str): The field for which the normalization parameters are being set.
        scale (float): The scale factor for normalization.
        shift (float): The shift value for normalization.
        symbol (str): The symbol representing the normalized quantity.
        units (str): The units of the normalized quantity.
        """
        self.dict[key+'scale']  = scale
        self.dict[key+'shift']  = shift
        self.dict[key+'symbol'] = symbol
        self.dict[key+'units']  = units

    def set(self, key, norm):
        """
        Normalize a specified key based on the provided normalization type.
        Use self.norm_help() for a list of available normalizations.

        Parameters:
        key (str): The key to be normalized (e.g., 'T', 'p', 'Wkin').
        norm (str): The type of normalization to apply. Available options include:
            - 'mus': Microseconds (µs)
            - 'vti/R': Time normalized by ion thermal velocity over major radius (t v_{ti}/R)
            - 'rho': Normalized to the minor radius (ρ)
            - 'x/rho': Normalized to the Larmor radius (ρ_L)
            - 'R-Rlcfs': Shift relative to the Last Closed Flux Surface (R - R_LCFS)
            - 'thermal velocity': Parallel velocities normalized by thermal velocity
            - 'eV': Energy in electron volts (eV)
            - 'MJ': Energy in megajoules (MJ)
            - 'beta': Pressure normalized by magnetic pressure (β)
            - 'Pa': Pressure in pascals (Pa)
            - 'temperatures': Normalizes all temperature components
            - 'fluid velocities': Normalizes both parallel electron and ion velocities
            - 'pressures': Normalizes all pressure components
            - 'energies': Normalizes all energy components
            - 'gradients': Normalizes gradients of specified quantities

        This method updates the normalization dictionary and log with the new normalization settings.
        """
        scale = 0
        ion = self.simulation.species['ion']

        #-- Time scale
        if norm == 'mus':
            scale  = 1e-6
            shift  = 0
            symbol = 't'
            units  = r'$\mu s$'
        elif norm in ['t v_{ti}/R', 't*vti/R', 'vti/R']:
            scale  = self.simulation.geom_param.R_axis / ion.vt
            shift  = 0
            symbol = r'$t v_{ti}/R$'
            units  = ''

        #-- Length scale
        elif norm in ['rho', 'minor radius']:
            scale  = self.simulation.geom_param.a_mid
            shift  = (self.simulation.geom_param.x_LCFS - self.simulation.geom_param.a_mid) / scale
            symbol = r'$r/a$'
            units  = ''
        elif norm in ['major radius']:
            scale  = self.simulation.geom_param.R_axis
            shift  = 0
            for spec in self.simulation.species.values():
                if key == 'gradlogn%s'%spec.nshort:
                    symbol = r'$R/L_{n%s}$'%spec.nshort
                elif key == 'gradlogT%s'%spec.nshort:
                    symbol = r'$R/L_{T%s}$'%spec.nshort
            units  = ''
        elif norm in ['x/rho', 'rho_L', 'Larmor radius']:
            scale  = ion.rho
            shift  = 0
            symbol = r'$%s/\rho_{0i}$' % key
            units  = ''
        elif norm in ['R-Rlcfs', 'LCFS shift', 'LCFS']:
            scale  = 1.0
            shift  = self.simulation.geom_param.x_LCFS
            symbol = r'$R-R_{LCFS}$'
            units  = 'm'
        elif norm in ['pi']:
            scale  = np.pi
            shift  = 0
            symbol = r'$z/\pi$'
            units  = ''
            
        #-- Wavenumber normalization
        elif norm in ['rho_i']:
            scale  = 1.0/self.simulation.species['ion'].rho
            shift  = 0
            symbol = r'$%s_{%s} \rho_{0i}$' %(key[0],key[-1])
            units  = ''
        elif norm in ['rho_e']:
            scale  = 1.0/self.simulation.species['elc'].rho
            shift  = 0
            symbol = r'$%s \rho_{0e}$' % key
            units  = ''
            
        #-- Velocity normalization
        elif norm in ['vt', 'thermal velocity']:
            for spec in self.simulation.species.values():
                if key == 'upar%s'%spec.nshort:
                    scale  = spec.vt
                    shift  = 0
                    symbol = r'$u_{\parallel %s}/v_{t0 %s}$'%(spec.nshort,spec.nshort)
                    units  = ''

        #-- Energy normalization
        elif norm in ['MJ','J']:
            scale = 1e6 if norm == 'MJ' else 1.0
            for spec in self.simulation.species.values():
                if key == 'Wkin%s'%spec.name[0]:
                    symbol = r'$W_{k,%s}$'%spec.name[0]
                if key == 'Wflu%s'%spec.name[0]:
                    symbol = r'$W_{f,%s}$'%spec.name[0]
                elif key == 'Wpot%s'%spec.name[0]:
                    symbol = r'$W_{p,%s}$'%spec.name[0]
                elif key == 'Wtot%s'%spec.name[0]:
                    symbol = r'$W_{%s}$'%spec.name[0]
            if key == 'Wkin':
                symbol = r'$W_{k}$'
            if key == 'Wflu':
                symbol = r'$W_{f}$'
            if key == 'Wpot':
                symbol = r'$W_{p}$'
            if key == 'Welf':
                symbol = r'$W_{E}$'
            if key == 'Wtot':
                symbol = r'$W$'
            shift = 0
            units = r'MJ/m$^3$' if norm == 'MJ' else r'J/m$^3$'

        #-- Temperature normalization
        elif norm == 'eV':
            for spec in self.simulation.species.values():
                if key == 'T%s'%spec.name[0]:
                    scale  = self.simulation.phys_param.eV / spec.m
                    symbol = r'$T_%s$'%spec.name[0]
                elif key == 'Tpar%s'%spec.name[0]:
                    scale  = self.simulation.phys_param.eV / spec.m
                    symbol = r'$T_{\parallel %s}$'%spec.name[0]
                elif key == 'Tperp%s'%spec.name[0]:
                    scale  = self.simulation.phys_param.eV / spec.m
                    symbol = r'$T_{\perp %s}$'%spec.name[0]
            shift = 0
            units = 'eV'
            if self.simulation.data_param.default_mom_type == 'M0' :
                scale /= self.phys_param.eV / 3.0

        #-- Preessure normalization
        elif norm == 'beta':
            mu0 = 4*np.pi*1e-7
            for spec in self.simulation.species.values():
                scale = 0.01*(self.simulation.geom_param.B0**2/(2*mu0)) * 1.0/spec.m
                if key == 'ppar%s'%spec.nshort:
                    symbol = r'$\beta_{\parallel %s}$'%spec.nshort
                elif key == 'pperp%s'%spec.nshort:
                    symbol = r'$\beta_{\perp %s}$'%spec.nshort
                elif key == 'p%s'%spec.nshort:
                    symbol = r'$\beta_{%s}$'%spec.nshort
            shift = 0
            units = r'$\%$'
            if self.simulation.data_param.default_mom_type == 'M0':
                scale /= self.simulation.phys_param.eV / 3.0
        elif norm == 'Pa':
            for spec in self.simulation.species.values():
                scale = 1/spec.m
                if key == 'ppar%s'%spec.nshort:
                    symbol = r'$p_{\parallel %s}$'%spec.nshort
                elif key == 'pperp%s'%spec.nshort:
                    symbol = r'$p_{\perp %s}$'%spec.nshort
                elif key == 'p%s'%spec.nshort:
                    symbol = r'$p_{%s}$'%spec.nshort
            shift = 0
            units = r'Pa'
            if self.simulation.data_param.default_mom_type == 'M0':
                scale /= self.simulation.phys_param.eV / 3.0
        
        #-- Current density normalization
        elif norm == 'MA':
            scale = 1e6
            symbol = r'$j_\parallel$'
            units = r'MA/m$^3$'
            shift = 0
        elif norm == 'kA':
            scale = 1e3
            symbol = r'$j_\parallel$'
            units = r'kA/m$^3$'
            shift = 0

        #-- Grouped normalization
        if key.lower() == 'temperatures':
            for spec in self.simulation.species.values():
                self.set(    'T%s'%spec.nshort, norm)
                self.set( 'Tpar%s'%spec.nshort, norm)
                self.set('Tperp%s'%spec.nshort, norm)

        elif key.lower() == 'fluid velocities':
            for spec in self.simulation.species.values():
                self.set('upar%s'%spec.nshort, norm)

        elif key.lower() == 'pressures':
            for spec in self.simulation.species.values():
                self.set(    'p%s'%spec.nshort, norm)
                self.set( 'ppar%s'%spec.nshort,  norm)
                self.set('pperp%s'%spec.nshort, norm)

        elif key.lower() == 'energies':
            self.set('Wkin',norm)
            self.set('Wflu',norm)
            self.set('Wpot',norm)
            self.set('Welf',norm)
            self.set('Wtot',norm)
            for spec in self.simulation.species.values():
                self.set('Wkin%s'%spec.nshort,norm)
                self.set('Wflu%s'%spec.nshort,norm)
                self.set('Wpot%s'%spec.nshort,norm)
                self.set('Wtot%s'%spec.nshort,norm)
        elif key.lower() == 'current':
            self.set('jpar',norm)
        elif key.lower() == 'gradients':
            self.set('gradlogn%s'%spec.nshort, norm)
            self.set('gradlogT%s'%spec.nshort, norm)
        else:
            #-- Apply normalization or handle unknown norm
            if scale != 0:
                self.change(key=key, scale=scale, shift=shift, symbol=symbol, units=units)
                self.norm_log.append(f"{key:<8} {norm:<16} {symbol:<20} {('['+units+']'):<8}")
            else:
                print(f"Warning: The normalization '{norm}' for '{key}'"+
                      " is not recognized. Please check the inputs or refer"+
                      " to the documentation for valid options:")
                self.help()
    
    def help(self):
        help_message = """
        Available Normalizations:
        
        1. **Time Normalizations**:
        - 'mus': Microseconds (µs)
        - 'vti/R': Time normalized by ion thermal velocity over major radius (t v_{ti}/R)
        
        2. **Length Normalizations**:
        - 'rho' or 'minor radius': Normalized to the minor radius (ρ)
        - 'x/rho' or 'rho_L' or 'Larmor radius': Normalized to the Larmor radius (ρ_L)
        - 'R-Rlcfs' or 'LCFS shift' or 'LCFS': Shift relative to 
            the Last Closed Flux Surface (R - R_LCFS)
        
        3. **Velocity Normalizations**:
        - 'thermal velocity' (when key is 'upare' or 'upari'): Parallel velocities 
            normalized by thermal velocity
        
        4. **Temperature Normalizations**:
        - 'eV': Energy in electron volts (eV), applicable to various temperature components:
            * 'Tpare': Parallel electron temperature (T_{\\parallel e})
            * 'Tperpe': Perpendicular electron temperature (T_{\\perp e})
            * 'Tpari': Parallel ion temperature (T_{\\parallel i})
            * 'Tperpi': Perpendicular ion temperature (T_{\\perp i})
        
        **Grouped Normalizations**:
        - 'temperatures': Normalizes all temperature components (Tpare, Tperpe, Tpari, Tperpi)
        - 'fluid velocities': Normalizes both parallel electron and ion velocities (upare, upari)

        Note: you can setup a custom normalization using the set_normalization routine, e.g.
                simulation.set_normalization(
                    key='ni', scale=ion.n0, shift=0, symbol=r'$n_i/n_0$', units=''
                    )
              sets a normalization of the ion density to n0
        """
        print(help_message)

    def info(self):
        print("Normalization Log:")
        print("Key     Normalization     Symbol               Units")
        print("------------------------------------------------------")
        for msg in self.norm_log:
            print(msg)
        print("------------------------------------------------------")