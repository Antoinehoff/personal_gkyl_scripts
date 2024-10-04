# DataParam.py
# This module defines the `DataParam` class, which handles the setup and management of various
# simulation data directories, file prefixes, and data fields for use in simulations. The class
# provides methods for configuring field dictionaries, file paths, and default units used in
# post-processing plasma simulation data. It is designed to accommodate both BiMaxwellian and 
# non-BiMaxwellian data.
import copy
class DataParam:
    def __init__(self, expdatadir='', g0simdir='', simname='', simdir='', 
                 fileprefix='', wkdir='', BiMaxwellian=True):
        self.expdatadir = expdatadir
        self.g0simdir = g0simdir
        self.simname = simname
        self.simdir = g0simdir+simdir
        self.wkdir = wkdir
        self.datadir = g0simdir + simdir + simname +'/' + wkdir
        self.fileprefix = self.datadir + fileprefix
        self.data_files_dict = {}
        self.BiMaxwellian = BiMaxwellian
        self.set_data_field_dict(BiMaxwellian=BiMaxwellian)

    def info(self):
        """
        Display the information of the directory parameters.
        """
        print(f"Directory Parameters:\n"
              f"  Experiment Data Directory (expdatadir): {self.expdatadir}\n"
              f"  G0 Simulation Directory (g0simdir): {self.g0simdir}\n"
              f"  Simulation Name (simname): {self.simname}\n"
              f"  Simulation Directory (simdir): {self.simdir}\n"
              f"  File Prefix (fileprefix): {self.fileprefix}\n"
              f"  Working Directory (wkdir): {self.wkdir}\n"
              f"  Data Directory (datadir): {self.datadir}\n")
        
    def set_data_field_dict(self,keys=[],files=[],comps=[],BiMaxwellian=True):
        '''
        This function set up the data field dictionary which indicates how each 
        possible scalar field can be found
        -file: gives the *.gkyl file specification
        -comp: the component to look for in the file
        -grids: the grid identificators 
        '''
        data_field_dict = {}
        gnames   = ['x','y','z','vpar','mu']
        # add electrostatic field info
        data_field_dict['phi'+'file'] = 'field'
        data_field_dict['phi'+'comp'] = 0
        data_field_dict['phi'+'gnames'] = gnames[0:3]
        
        # add moments info        
        keys  = ['n','upar','Tpar','Tperp','ppar','pperp']
        sname = ['ion','elc']
        for s_ in sname:
            shortname = s_[0]
            if BiMaxwellian:
                comps  = [0,1,2,3,0,0]
                prefix = 6*[s_+'_BiMaxwellianMoments']
            else:
                comps  = [0,0,0,0,0,0]
                prefix = [s_+'_M0',s_+'_M1',s_+'_M2par',s_+'_M2perp',s_+'_M0',s_+'_M0']
            for i in range(len(keys)):
                data_field_dict[keys[i]+shortname+'file']   = prefix[i]
                data_field_dict[keys[i]+shortname+'comp']   = comps[i]
                data_field_dict[keys[i]+shortname+'gnames'] = gnames[0:3]

            # add distribution functions
            data_field_dict['f'+shortname+'file'] = s_
            data_field_dict['f'+shortname+'comp'] = 0
            data_field_dict['f'+shortname+'gnames'] = gnames

        self.data_files_dict = data_field_dict
        
    @staticmethod
    def get_default_units_dict():
        default_units_dict = {}
        # We define the fields that we are able to plot and load
        default_qttes = [
            ['x', r'$x$', 'm'],
            ['y', r'$y$', 'm'],
            ['z', r'$z$', 'm'],
            ['vpar', r'$v_\parallel$', 'm/s'],
            ['mu', r'$\mu$', 'J/T'],
            ['t', r'$t$', 's'],
            ['phi', r'$\phi$', 'V']
            ]
        #-add routinely other quantities of interest
        for s_ in ['e','i']:
            # distribution functions
            default_qttes.append(['f%s'%(s_), r'$f_%s$'%(s_), '[f]'])
            # densities
            default_qttes.append(['n%s'%(s_), r'$n_%s$'%(s_), r'm$^{-3}$'])
            # parallel velocities
            default_qttes.append(['upar%s'%(s_), r'$u_{\parallel %s}$'%(s_), 'm/s'])
            #parallel and perpendicular temperatures
            default_qttes.append(['Tpar%s'%(s_), r'$T_{\parallel %s}$'%(s_), 'J/kg'])
            default_qttes.append(['Tperp%s'%(s_), r'$T_{\perp %s}$'%(s_), 'J/kg'])
            #parallel and perpendicular pressures
            default_qttes.append(['ppar%s'%(s_), r'$p_{\parallel %s}$'%(s_), r'J/kg/m$^{3}$'])
            default_qttes.append(['pperp%s'%(s_), r'$p_{\perp %s}$'%(s_), r'J/kg/m$^{3}$'])
            for c_ in ['x','y','z']:
                # particle fluxes
                default_qttes.append(['pflux%s%s'%(c_,s_),r'$\Gamma_{%s %s}$'%(c_,s_),r's$^{-1}$m$^{-2}$'])
                # heat fluxes
                default_qttes.append(['qflux%s%s'%(c_,s_),r'$Q_{%s %s}$'%(c_,s_),r'J kg$^{-1}$ s$^{-1}$m$^{-2}$'])
                
        names   = [default_qttes[i][0] for i in range(len(default_qttes))]
        symbols = {default_qttes[i][0]: default_qttes[i][1] for i in range(len(names))}
        units   = {default_qttes[i][0]: default_qttes[i][2] for i in range(len(names))}
        for key in names:
            default_units_dict[key+'scale']  = 1.0
            default_units_dict[key+'shift']  = 0.0
            default_units_dict[key+'symbol'] = symbols[key]
            default_units_dict[key+'units']  = units[key]
        # and initialize the normalization
        return copy.copy(default_units_dict)
