# DataParam.py
# This module defines the `DataParam` class, which handles the setup and management of various
# simulation data directories, file prefixes, and data fields for use in simulations. The class
# provides methods for configuring field dictionaries, file paths, and default units used in
# post-processing plasma simulation data. It is designed to accommodate both BiMaxwellian and 
# non-BiMaxwellian data.
import copy
import numpy as np
class DataParam:
    def __init__(self, expdatadir='', g0simdir='', simname='', simdir='', 
                 fileprefix='', wkdir='', BiMaxwellian=True, species = {}):
        self.expdatadir = expdatadir
        self.g0simdir = g0simdir
        self.simname = simname
        self.simdir = g0simdir + simdir
        self.wkdir = wkdir
        self.datadir = g0simdir + simdir + simname +'/' + wkdir
        self.fileprefix = self.datadir + fileprefix
        self.data_files_dict = {}
        self.BiMaxwellian = BiMaxwellian
        self.set_data_field_dict(BiMaxwellian=BiMaxwellian, species=species)
        # We set here an array where all quantities that does not depend on species are stored. 
        # This helps in the treatment of input in the plot functions
        self.spec_undep_quantities = ['phi','Apar','b_x','b_y','b_z','Jacobian','Bmag','Wtot']
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
        
    def set_data_field_dict(self,keys=[],files=[],comps=[],BiMaxwellian=True,species={}):
        '''
        This function set up the data field dictionary which indicates how each 
        possible scalar field can be found
        -file: gives the *.gkyl file specification
        -comp: the component to look for in the file
        -grids: the grid identificators 
        '''
        data_field_dict = {}
        gnames   = ['x','y','z','vpar','mu']

        # add equilibrium info
        # Magnetic field amplitude
        data_field_dict['Bmag'+'file']   = 'bmag'
        data_field_dict['Bmag'+'comp']   = 0
        data_field_dict['Bmag'+'gnames'] = gnames[0:3]

        # normalized b field
        for i_ in range(3):
            data_field_dict['b_'+gnames[i_]+'file']   = 'b_i'
            data_field_dict['b_'+gnames[i_]+'comp']   = i_
            data_field_dict['b_'+gnames[i_]+'gnames'] = gnames[0:3]

        # Jacobian
        data_field_dict['Jacobian'+'file']   = 'jacobgeo'
        data_field_dict['Jacobian'+'comp']   = 0
        data_field_dict['Jacobian'+'gnames'] = gnames[0:3]

        # add electrostatic field info
        data_field_dict['phi'+'file'] = 'field'
        data_field_dict['phi'+'comp'] = 0
        data_field_dict['phi'+'gnames'] = gnames[0:3]
        
        # add moments info        
        keys  = ['n','upar','Tpar','Tperp','ppar','pperp']
        for spec in species.values():
            s_        = spec.name
            shortname = spec.nshort
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
    def get_default_units_dict(species):
        #-Here we define the default unit dictionary which is meant to gives 
        # the symbol, units, scale, shift, and composition 
        # of a given quantity. 
        # The quantities are identified following identification convention that is used for using the plot routines
        # ('x' for x grid, 'phi' or ES field, 'n' for density, 'Tperp' for perp temperature etc.)
        # E.g.  for the z grid, we want default_units_dict['zsymbol'] = r'$z$'
        #       for the field phi, we want default_units_dict['phiunit'] = 'V'
        #       etc.
        default_units_dict = {}
        #-First we define the fields that we are able to plot and load
        # the data is organized as follow [identification, symbol, units, composition, receipe]
        # a composition is appended afterwards, for the following, composition = [identification]
        # for composed quantities:
        # e.g. perpendicular electron pressure, composition = [ne, Tperpe], receipe = composition[0] * composition[1]
        default_qttes = [
            ['x', r'$x$', 'm'],
            ['y', r'$y$', 'm'],
            ['z', r'$z$', ''],
            ['vpar', r'$v_\parallel$', 'm/s'],
            ['mu', r'$\mu$', 'J/T'],
            ['t', r'$t$', 's'],
            ['phi', r'$\phi$', 'V'],
            ['b_x', r'$b_x$', ''],
            ['b_y', r'$b_y$', ''],
            ['b_z', r'$b_z$', ''],
            ['Jacobian', r'$J$', '[Jacobian]'],
            ['Bmag', r'$B$','T']
            ]
        # add routinely other quantities of interest
        for spec in species.values():
            s_ = spec.nshort
            # distribution functions
            default_qttes.append(['f%s'%(s_), r'$f_%s$'%(s_), '[f]'])
            # densities
            default_qttes.append(['n%s'%(s_), r'$n_%s$'%(s_), r'm$^{-3}$'])
            # parallel velocities
            default_qttes.append(['upar%s'%(s_), r'$u_{\parallel %s}$'%(s_), 'm/s'])
            #parallel and perpendicular temperatures
            default_qttes.append(['Tpar%s'%(s_), r'$T_{\parallel %s}$'%(s_), 'J/kg'])
            default_qttes.append(['Tperp%s'%(s_), r'$T_{\perp %s}$'%(s_), 'J/kg'])
        #-The above defined fields are all simple quantities in the sense that 
        # composition=[identification] and so receipe = composition[0]
        def identity(gdata_list):
            return gdata_list[0].get_values()
        # so we can define the compositions in one line here
        for i in range(len(default_qttes)):
            default_qttes[i].append([default_qttes[i][0]])
            default_qttes[i].append(identity)

        #-We define now composed quantities as pressures and fluxes 
        # for each species present in the simulation
        for spec in species.values():
            s_ = spec.nshort

            #locally normalized parallel velocity   
            name       = 'spar%s'%(s_)
            symbol     = r'$u_{\parallel %s}/v_{t %s}$'%(s_,s_)
            units      = ''
            field2load = ['upar%s'%(s_),'Tpar%s'%(s_),'Tperp%s'%(s_)]
            def receipe_spars(gdata_list):
                upar = gdata_list[0].get_values()
                Tom  = (gdata_list[1].get_values() + 2.0*gdata_list[2].get_values())/3.0
                vt   = np.sqrt(2*Tom)
                return upar/vt
            default_qttes.append([name,symbol,units,field2load,receipe_spars])      

            #total temperature
            name       = 'T%s'%(s_)
            symbol     = r'$T_{%s}$'%(s_)
            units      = 'J/kg'
            field2load = ['Tpar%s'%(s_),'Tperp%s'%(s_)]
            def receipe_Ttots(gdata_list):
                return (gdata_list[0].get_values() + 2.0*gdata_list[1].get_values())/3.0
            default_qttes.append([name,symbol,units,field2load,receipe_Ttots])
            
            #total energy speciewise: Ws = int dv3 1/2 ms vpar^2 + mus B - qs phi
            name       = 'W%s'%(s_)
            symbol     = r'$W_{%s}$'%(s_) 
            units      = r'J/m$^3$'
            field2load = ['n%s'%s_,'Tpar%s'%(s_),'Tperp%s'%(s_),'phi']
            # We need the mass and the charge to converte J/kg (temp) and V (phi) in Joules
            def receipe_Ws(gdata_list,q=spec.q,m=spec.m):
                dens = gdata_list[0].get_values()
                Ttot = (gdata_list[1].get_values() + 2.0*gdata_list[2].get_values())/3.0
                qphi = q*gdata_list[3].get_values()
                return dens*(m*Ttot + qphi)
            default_qttes.append([name,symbol,units,field2load,receipe_Ws])

            #total pressure speciewise
            name = 'p%s'%(s_)
            symbol = r'$p_{%s}$'%(s_)
            units = 'J/kg/m$^{3}$'
            field2load = ['n%s'%(s_),'Tpar%s'%(s_),'Tperp%s'%(s_)]
            def receipe_ptots(gdata_list):
                return gdata_list[0].get_values()*(gdata_list[1].get_values() + 2.0*gdata_list[2].get_values())/3.0
            default_qttes.append([name,symbol,units,field2load,receipe_ptots])

            #parallel pressures
            name       = 'ppar%s'%(s_)
            symbol     = r'$p_{\parallel,%s}$'%(s_)
            units      = 'J/kg/m$^{3}$'
            field2load = ['n%s'%(s_),'Tpar%s'%(s_)]
            def receipe_ppars(gdata_list):
                return gdata_list[0].get_values()*gdata_list[1].get_values()/3.0
            default_qttes.append([name,symbol,units,field2load,receipe_ppars])

            #perpendicular pressures
            name       = 'pperp%s'%(s_)
            symbol     = r'$p_{\perp,%s}$'%(s_)
            units      = 'J/kg/m$^{3}$'
            field2load = ['n%s'%(s_),'Tperp%s'%(s_)]
            def receipe_pperps(gdata_list):
                return gdata_list[0].get_values()*gdata_list[1].get_values()*2.0/3.0     
            default_qttes.append([name,symbol,units,field2load,receipe_pperps])

            #normalized pressure beta
            name = 'beta%s'%(s_)
            symbol = r'$\beta_{%s}$'%(s_)
            units = '%'
            field2load = ['n%s'%(s_),'Tpar%s'%(s_),'Tperp%s'%(s_),'Bmag']
            def receipe_betas(gdata_list):
                mu0 = 4.0*np.pi*1e-7
                dens = gdata_list[0].get_values()
                Ttot = (gdata_list[1].get_values() + 2.0*gdata_list[2].get_values())/3.0*spec.m
                Bmag = gdata_list[3].get_values()
                return 100*dens*Ttot*2*mu0/np.power(Bmag,2)
            default_qttes.append([name,symbol,units,field2load,receipe_betas])

            #- The following are vector fields quantities that we treat component wise
            directions = ['x','y','z'] #directions array
            for i_ in range(len(directions)):
                ci_ = directions[i_] # direction of the flux component
                cj_ = directions[np.mod(i_+1,3)] # direction coord + 1
                ck_ = directions[np.mod(i_+2,3)] # direction coord + 2

                # particle flux
                name       = 'pflux%s%s'%(ci_,s_)
                symbol     = r'$\Gamma_{%s %s}$'%(ci_,s_)
                units      = r's$^{-1}$m$^{-2}$'
                field2load = ['n%s'%s_,'phi','b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian']
                # The receipe depends on the direction 
                # because of the phi derivative
                def receipe_pfluxs(gdata_list,i=i_):
                    density = gdata_list[0].get_values()
                    phi     = gdata_list[1].get_values()
                    b_j     = gdata_list[2].get_values()
                    b_k     = gdata_list[3].get_values()
                    Bmag    = gdata_list[4].get_values()
                    Jacob   = gdata_list[5].get_values()
                    j       = np.mod(i+1,3)
                    k       = np.mod(i+2,3)
                    grids   = gdata_list[0].get_grid()
                    jgrid   = grids[j][:-1]
                    kgrid   = grids[k][:-1]
                    dphidj  = np.gradient(phi, jgrid, axis=j)
                    dphidk  = np.gradient(phi, kgrid, axis=k)
                    return -density*(dphidj*b_k - dphidk*b_j)/Jacob/Bmag
                
                default_qttes.append([name,symbol,units,field2load,receipe_pfluxs])

                # heat fluxes
                name       = 'hflux%s%s'%(ci_,s_)
                symbol     = r'$Q_{%s %s}$'%(ci_,s_)
                units       = r'J s$^{-1}$m$^{-2}$'
                field2load = ['n%s'%s_,'Tpar%s'%s_,'Tperp%s'%s_,
                              'phi','b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian']
                # The receipe depends on the direction 
                # because of the phi derivative and on the species 
                # (temperature from J/kg to J)
                def receipe_hfluxs(gdata_list,i=i_,m=spec.m):
                    density = gdata_list[0].get_values()
                    Ttot    = (gdata_list[1].get_values() + 2.*gdata_list[2].get_values())
                    Ttot    = m*Ttot/3.0 # 1/3 normalization (Tpar+2Tperp) and conversion to joules
                    phi     = gdata_list[3].get_values()
                    b_j     = gdata_list[4].get_values()
                    b_k     = gdata_list[5].get_values()
                    Bmag    = gdata_list[6].get_values()
                    Jacob   = gdata_list[7].get_values()
                    j       = np.mod(i+1,3)
                    k       = np.mod(i+2,3)
                    grids   = gdata_list[0].get_grid()
                    jgrid   = grids[j][:-1]
                    kgrid   = grids[k][:-1]
                    dphidj  = np.gradient(phi, jgrid, axis=j)
                    dphidk  = np.gradient(phi, kgrid, axis=k)
                    return -density*Ttot*(dphidj*b_k - dphidk*b_j)/Jacob/Bmag
                default_qttes.append([name,symbol,units,field2load,receipe_hfluxs])

        # Species summed quantities
        #total energy : \sum_s W_s = int dv3 1/2 ms vpar^2 + mus B - qs phi
        name       = 'Wtot'
        symbol     = r'$W$'
        units       = r'J/m$^3$'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('n%s'%s_)
            field2load.append('Tpar%s'%(s_))
            field2load.append('Tperp%s'%(s_))
            field2load.append('phi')
        def receipe_Wtot(gdata_list,species=species):
            fout = 0.0
            k    = 0
            for spec in species.values():
                fout += receipe_Ws(gdata_list[0+k:4+k],q=spec.q,m=spec.m)
                k += 4
            return fout 
        default_qttes.append([name,symbol,units,field2load,receipe_Wtot])
        
        ## We format everything so that it fits in one dictionary
        names   = [default_qttes[i][0] for i in range(len(default_qttes))]
        symbols = {default_qttes[i][0]: default_qttes[i][1] for i in range(len(names))}
        units   = {default_qttes[i][0]: default_qttes[i][2] for i in range(len(names))}
        compo   = {default_qttes[i][0]: default_qttes[i][3] for i in range(len(names))}
        receipe = {default_qttes[i][0]: default_qttes[i][4] for i in range(len(names))}
        for key in names:
            default_units_dict[key+'scale']    = 1.0
            default_units_dict[key+'shift']    = 0.0
            default_units_dict[key+'symbol']   = symbols[key]
            default_units_dict[key+'units']    = units[key]
            default_units_dict[key+'compo']    = compo[key]
            default_units_dict[key+'receipe']  = receipe[key]

        # Return a copy of the units dictionary
        return copy.copy(default_units_dict)
