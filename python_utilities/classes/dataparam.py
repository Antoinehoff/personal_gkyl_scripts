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
            ['ky', r'$k_y$', '1/m'],
            ['wavelen', r'$\lambda$', 'm'],
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

        #-Drift velocities
        #- The following are vector fields quantities that we treat component wise
        directions = ['x','y','z'] #directions array
        for i_ in range(len(directions)):
            ci_ = directions[i_] # direction of the flux component
            cj_ = directions[np.mod(i_+1,3)] # direction coord + 1
            ck_ = directions[np.mod(i_+2,3)] # direction coord + 2

            # ExB velocity
            name       = 'vExB_%s'%(ci_)
            symbol     = r'$u_{E,%s}$'%(ci_)
            units      = r'm/s'
            field2load = ['b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian','phi']
            # The receipe depends on the direction 
            # because of the phi derivative
            def receipe_vExB(gdata_list,i=i_):
                b_j     = gdata_list[0].get_values()
                b_k     = gdata_list[1].get_values()
                Bmag    = gdata_list[2].get_values()
                Jacob   = gdata_list[3].get_values()
                grids   = gdata_list[0].get_grid()
                j       = np.mod(i+1,3)
                k       = np.mod(i+2,3)
                jgrid   = grids[j][:-1]
                kgrid   = grids[k][:-1]
                phi     = gdata_list[4].get_values()
                dphidj  = np.gradient(phi, jgrid, axis=j)
                dphidk  = np.gradient(phi, kgrid, axis=k)
                return -(dphidj*b_k - dphidk*b_j)/Jacob/Bmag
            default_qttes.append([name,symbol,units,field2load,receipe_vExB])

            # ExB shearing rate
            name       = 'sExB_%s'%(ci_)
            symbol     = r'$s_{E,%s}$'%(ci_)
            units      = r'1/s'
            field2load = ['b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian','phi']
            # The receipe depends on the direction 
            # because of the phi derivative
            def receipe_sExB(gdata_list,i=i_):
                vExB  = receipe_vExB(gdata_list,i=i)
                grids = gdata_list[0].get_grid()
                igrid = grids[i][:-1]
                sExB  = np.gradient(vExB, igrid, axis=i)
                return sExB/vExB
            default_qttes.append([name,symbol,units,field2load,receipe_sExB])

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
            units      = 'J/kg' # T is stored as T/m in gkeyll
            field2load = ['Tpar%s'%(s_),'Tperp%s'%(s_)]
            def receipe_Ttots(gdata_list):
                return (gdata_list[0].get_values() + 2.0*gdata_list[1].get_values())/3.0
            default_qttes.append([name,symbol,units,field2load,receipe_Ttots])

            #Kinetic energy density speciewise: Wkins = int dv3 1/2 ms vpar^2 + mus B
            name       = 'Wkin%s'%(s_)
            symbol     = r'$W_{k,%s}$'%(s_) 
            units      = r'J/m$^3$'
            field2load = ['n%s'%s_,'Tpar%s'%(s_),'Tperp%s'%(s_)]
            def receipe_Wkins(gdata_list,m=spec.m):
                dens = gdata_list[0].get_values()
                Ttot = receipe_Ttots(gdata_list[1:3])
                return dens*m*Ttot
            default_qttes.append([name,symbol,units,field2load,receipe_Wkins])

            #Fluid kinetic energy density speciewise: Wflu = 1/2 m*n*upar^2
            name       = 'Wflu%s'%(s_)
            symbol     = r'$W_{f,%s}$'%(s_) 
            units      = r'J/m$^3$'
            field2load = ['n%s'%s_,'upar%s'%(s_)]
            def receipe_Wflus(gdata_list,m=spec.m):
                dens = gdata_list[0].get_values()
                upar = gdata_list[1].get_values()
                return dens*m*np.power(upar,2)/2.0
            default_qttes.append([name,symbol,units,field2load,receipe_Wflus])

            #Potential energy density speciewise: Wpots = qs phi
            name       = 'Wpot%s'%(s_)
            symbol     = r'$W_{p,%s}$'%(s_) 
            units      = r'J/m$^3$'
            field2load = ['phi','n%s'%s_]
            def receipe_Wpots(gdata_list,q=spec.q):
                qphi = q*gdata_list[0].get_values()
                dens = gdata_list[1].get_values()
                return dens*qphi
            default_qttes.append([name,symbol,units,field2load,receipe_Wpots])

            #total energy density speciewise: Ws = Wkins + Wflu + Wpots
            name       = 'Wtot%s'%(s_)
            symbol     = r'$W_{%s}$'%(s_) 
            units      = r'J/m$^3$'
            field2load = ['phi','n%s'%s_,'upar%s'%(s_),'Tpar%s'%(s_),'Tperp%s'%(s_)]
            # We need the mass and the charge to converte J/kg (temp) and V (phi) in Joules
            def receipe_Ws(gdata_list,q=spec.q,m=spec.m):
                qphi = q*gdata_list[0].get_values()
                dens = gdata_list[1].get_values()
                upar = gdata_list[2].get_values()
                Ttot = receipe_Ttots(gdata_list[3:5])
                return dens*(m*np.power(upar,2)/2 + m*Ttot + qphi)
            default_qttes.append([name,symbol,units,field2load,receipe_Ws])

            #total pressure speciewise
            name = 'p%s'%(s_)
            symbol = r'$p_{%s}$'%(s_)
            units = 'J/kg/m$^{3}$'
            field2load = ['n%s'%(s_),'Tpar%s'%(s_),'Tperp%s'%(s_)]
            def receipe_ptots(gdata_list):
                Ttot = receipe_Ttots(gdata_list[1:3])
                return gdata_list[0].get_values()*Ttot
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
            units = r'$\%$'
            field2load = ['n%s'%(s_),'Tpar%s'%(s_),'Tperp%s'%(s_),'Bmag']
            def receipe_betas(gdata_list,m=spec.m):
                mu0 = 4.0*np.pi*1e-7
                dens = gdata_list[0].get_values()
                Ttot = receipe_Ttots(gdata_list[1:3])
                Bmag = gdata_list[3].get_values()
                return 100 * dens * m*Ttot* 2*mu0/np.power(Bmag,2)
            default_qttes.append([name,symbol,units,field2load,receipe_betas])

            #- The following are vector fields quantities that we treat component wise
            directions = ['x','y','z'] #directions array
            for i_ in range(len(directions)):
                ci_ = directions[i_] # direction of the flux component
                cj_ = directions[np.mod(i_+1,3)] # direction coord + 1
                ck_ = directions[np.mod(i_+2,3)] # direction coord + 2

                # gradBxB velocity
                name       = 'gradBxB_%s%s'%(ci_,s_)
                symbol     = r'$u_{\nabla B,%s %s}$'%(ci_,s_)
                units      = r'm/s'
                field2load = ['b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian','Tperp%s'%s_]
                def receipe_vgB(gdata_list,i=i_,q=spec.q,m=spec.m):
                    b_j     = gdata_list[0].get_values()
                    b_k     = gdata_list[1].get_values()
                    Bmag    = gdata_list[2].get_values()
                    Jacob   = gdata_list[3].get_values()
                    Tperp   = gdata_list[4].get_values()*m
                    grids   = gdata_list[4].get_grid()
                    j       = np.mod(i+1,3)
                    k       = np.mod(i+2,3)
                    jgrid   = grids[j][:-1]
                    kgrid   = grids[k][:-1]
                    dBdj  = np.gradient(Bmag, jgrid, axis=j)
                    dBdk  = np.gradient(Bmag, kgrid, axis=k)
                    return Tperp*(b_j*dBdk - b_k*dBdj)/Jacob/Bmag/q
                default_qttes.append([name,symbol,units,field2load,receipe_vgB])

                # ExB particle flux
                name       = 'ExB_pflux_%s%s'%(ci_,s_)
                symbol     = r'$\Gamma_{E%s,%s}$'%(ci_,s_)
                units      = r's$^{-1}$m$^{-2}$'
                field2load = ['b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian',
                              'phi','n%s'%s_,]
                # The receipe depends on the direction 
                # because of the phi derivative
                def receipe_ExB_pflux_s(gdata_list,i=i_):
                    vE      = receipe_vExB(gdata_list,i=i)
                    density = gdata_list[5].get_values()
                    return density*vE
                
                default_qttes.append([name,symbol,units,field2load,receipe_ExB_pflux_s])

                # ExB heat fluxes
                name       = 'ExB_hflux_%s%s'%(ci_,s_)
                symbol     = r'$Q_{E%s,%s}$'%(ci_,s_)
                units       = r'J s$^{-1}$m$^{-2}$'
                field2load = ['b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian',
                              'phi','n%s'%s_,'Tpar%s'%s_,'Tperp%s'%s_,]
                # The receipe depends on the direction 
                # because of the phi derivative and on the species 
                # (temperature from J/kg to J)
                def receipe_ExB_hflux_s(gdata_list,i=i_,m=spec.m):
                    vE = receipe_vExB(gdata_list,i=i)
                    density = gdata_list[5].get_values()
                    Ttot    = receipe_Ttots(gdata_list[6:8])
                    return density * m*Ttot * vE
                default_qttes.append([name,symbol,units,field2load,receipe_ExB_hflux_s])

                # gradB particle flux
                name       = 'gradB_pflux_%s%s'%(ci_,s_)
                symbol     = r'$\Gamma_{\nabla B%s,%s}$'%(ci_,s_)
                units      = r's$^{-1}$m$^{-2}$'
                field2load = ['b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian',
                              'phi','n%s'%s_,]
                # The receipe depends on the direction 
                # because of the phi derivative
                def receipe_gradB_pflux_s(gdata_list,i=i_,q=spec.q,m=spec.m):
                    vgB     = receipe_vgB(gdata_list,i=i,q=q,m=m)
                    density = gdata_list[5].get_values()
                    return density*vgB
                default_qttes.append([name,symbol,units,field2load,receipe_gradB_pflux_s])
                
                # gradB heat flux
                name       = 'gradB_hflux_%s%s'%(ci_,s_)
                symbol     = r'$Q_{\nabla B%s,%s}$'%(ci_,s_)
                units      = r'J s$^{-1}$m$^{-2}$'
                field2load = ['b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian',
                              'phi','n%s'%s_,'Tpar%s'%s_,'Tperp%s'%s_,]
                # The receipe depends on the direction 
                # because of the phi derivative
                def receipe_gradB_hflux_s(gdata_list,i=i_,q=spec.q,m=spec.m):
                    vgB     = receipe_vgB(gdata_list,i=i,q=q,m=m)
                    density = gdata_list[5].get_values()
                    Ttot    = receipe_Ttots(gdata_list[6:8])
                    #T is converted to joules by mass
                    return density * m*Ttot * vgB
                default_qttes.append([name,symbol,units,field2load,receipe_gradB_hflux_s])

        # Species independent quantities
        #charge density
        name       = 'qdens'
        symbol     = r'$\sum_s q_{s}n_s$'
        units       = r'C/m$^3$'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('n%s'%s_)
        def receipe_qdens(gdata_list,species=species):
            fout = 0.0
            # add species dependent energies
            k = 0
            for spec in species.values():
                fout += spec.q*gdata_list[0+k].get_values()
                k    += 1
            return fout
        default_qttes.append([name,symbol,units,field2load,receipe_qdens])

        #ion-electron temperature ratio
        name       = 'Tratio'
        symbol     = r'$T_{i}/T_e$'
        units      = '' # T is stored as T/m in gkeyll
        field2load = ['Tpare','Tperpe','Tpari','Tperpi']
        def receipe_Tratio(gdata_list,species=species):
            Te = (gdata_list[0].get_values() + 2.0*gdata_list[1].get_values())/3.0
            Ti = (gdata_list[2].get_values() + 2.0*gdata_list[3].get_values())/3.0
            me = species['elc'].m
            mi = species['ion'].m
            return Ti*mi/(Te*me)
        default_qttes.append([name,symbol,units,field2load,receipe_Tratio])

        #parallel current density
        name       = 'jpar'
        symbol     = r'$\sum_s j_{\parallel s}$'
        units       = r'A/m$^3$'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('n%s'%s_)
            field2load.append('upar%s'%s_)
        def receipe_jpar(gdata_list,species=species):
            fout = 0.0
            # add species dependent energies
            k = 0
            for spec in species.values():
                fout += spec.q*gdata_list[0+k].get_values()*gdata_list[1+k].get_values()
                k    += 2
            return fout
        default_qttes.append([name,symbol,units,field2load,receipe_jpar])

        #total energy : \sum_s W_kins = \sum_s int dv3 1/2 ms vpar^2 + mus B
        name       = 'Wkin'
        symbol     = r'$W_k$'
        units      = r'J/m$^3$'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('n%s'%s_)
            field2load.append('Tpar%s'%(s_))
            field2load.append('Tperp%s'%(s_))
        def receipe_Wkin(gdata_list,species=species):
            fout = 0.0
            k    = 0
            for spec in species.values():
                fout += receipe_Wkins(gdata_list[0+k:3+k],m=spec.m)
                k += 3
            return fout 
        default_qttes.append([name,symbol,units,field2load,receipe_Wkin])

        #total fluid kinetic energy : \sum_s W_flus
        name       = 'Wflu'
        symbol     = r'$W_f$'
        units      = r'J/m$^3$'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('n%s'%s_)
            field2load.append('upar%s'%(s_))
        def receipe_Wflu(gdata_list,species=species):
            fout = 0.0
            k    = 0
            for spec in species.values():
                fout += receipe_Wflus(gdata_list[0+k:2+k],m=spec.m)
                k += 2
            return fout 
        default_qttes.append([name,symbol,units,field2load,receipe_Wflu])

        #total potential energy : \sum_s W_pots = qs phi
        name       = 'Wpot'
        symbol     = r'$W_p$'
        units      = r'J/m$^3$'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('phi')
            field2load.append('n%s'%s_)
        def receipe_Wpot(gdata_list,species=species):
            fout = 0.0
            k    = 0
            for spec in species.values():
                fout += receipe_Wpots(gdata_list[0+k:2+k],q=spec.q)
                k += 2
            return fout 
        default_qttes.append([name,symbol,units,field2load,receipe_Wpot])
        
        ## EM field related data
        directions = ['x','y','z'] #directions array
        for i_ in range(len(directions)):
            ci_ = directions[i_] # direction of the component
            # Electric field i-th component
            name       = 'E%s'%(ci_)
            symbol     = r'$E_{%s}$'%(ci_)
            units      = r'V/m'
            field2load = ['phi']
            # The receipe depends on the direction 
            # because of the phi derivative
            def receipe_Ei(gdata_list,i=i_):
                phi     = gdata_list[0].get_values()
                grids   = gdata_list[0].get_grid()
                grid    = grids[i][:-1]
                return -np.gradient(phi, grid, axis=i)
            default_qttes.append([name,symbol,units,field2load,receipe_Ei])

        #electric field energy Welc = 1/2 eps0 |E|^2
        name       = 'Welf'
        symbol     = r'$W_{E}$'
        units      = r'J/m$^3$'
        field2load = ['phi']
        # The receipe depends on the direction 
        # because of the phi derivative
        def receipe_Welf(gdata_list):
            eps0 = 8.854e-12 # Vacuum permittivity in F⋅m−1
            fout = 0.0
            # compute the norm of the electric field
            for i_ in range(3):
                fout += np.power(receipe_Ei(gdata_list,i=i_),2)
            # multiply by 1/2 eps0
            fout *= 0.5*eps0
            return fout
        default_qttes.append([name,symbol,units,field2load,receipe_Welf])

        #total energy : \sum_s W_s = int dv3 1/2 ms vpar^2 + mus B - qs ns phi
        name       = 'Wtot'
        symbol     = r'$W_{tot}$'
        units       = r'J/m$^3$'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('phi')
            field2load.append('n%s'%s_)
            field2load.append('upar%s'%(s_))
            field2load.append('Tpar%s'%(s_))
            field2load.append('Tperp%s'%(s_))
        def receipe_Wtot(gdata_list,species=species):
            fout = 0.0
            # add species dependent energies
            k = 0
            for spec in species.values():
                fout += receipe_Ws(gdata_list[0+k:5+k],q=spec.q,m=spec.m)
                k += 5

            # add the EM energy
            fout += receipe_Welf(gdata_list[0:1])

            return fout
        default_qttes.append([name,symbol,units,field2load,receipe_Wtot])

        #total ExB heat flux: Q_ExB = \sum_s Q_ExB_s
        directions = ['x','y','z'] #directions array
        for i_ in range(len(directions)):
            ci_ = directions[i_] # direction of the flux component
            cj_ = directions[np.mod(i_+1,3)] # direction coord + 1
            ck_ = directions[np.mod(i_+2,3)] # direction coord + 2
            name       = 'ExB_hflux_%s'%(ci_)
            symbol     = r'$Q_{%s}$'%(ci_)
            units      = r'J s$^{-1}$m$^{-2}$'
            field2load = []
            for spec in species.values():
                s_ = spec.nshort
                field2load.append('b_%s'%cj_)
                field2load.append('b_%s'%ck_)
                field2load.append('Bmag')
                field2load.append('Jacobian')
                field2load.append('phi')
                field2load.append('n%s'%s_)
                field2load.append('Tpar%s'%s_)
                field2load.append('Tperp%s'%s_)
            def receipe_hflux(gdata_list,i=i_,species=species):
                fout = 0.0
                # add species dependent energies
                k = 0
                for spec in species.values():
                    # fout += receipe_gradB_hflux_s(gdata_list[0+k:8+k], i=i, q=spec.q, m=spec.m)
                    fout += receipe_ExB_hflux_s(gdata_list[0+k:8+k], i=i, m=spec.m)
                    k+= 8
                return fout
            default_qttes.append([name,symbol,units,field2load,receipe_hflux]) 
        #-------------- END of the new diagnostics definitions
        
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
