import copy
import numpy as np
import os
from ..tools import pgkyl_interface as pgkyl_

class DataParam:
    """
    DataParam
    ---------
    Manages the setup and configuration of simulation data directories, file prefixes, and data fields.

    Attributes:
    -----------
    - expdatadir (str): Directory for experimental data.
    - g0simdir (str): Directory for G0 simulation data.
    - simname (str): Name of the simulation.
    - simdir (str): Directory for simulation data.
    - fileprefix (str): Prefix for data files.
    - wkdir (str): Working directory.
    - datadir (str): Directory for data storage.
    - data_files_dict (dict): Dictionary mapping data fields to file specifications.
    - BiMaxwellian (bool): Flag indicating if BiMaxwellian moments are used.
    
    Methods:
    --------
    - __init__: Initializes the DataParam object with the required parameters.
    - info: Displays the information of the directory parameters.
    - set_data_field_dict: Sets up the data field dictionary for various scalar fields.
    - get_default_units_dict: Returns the default units dictionary for various quantities.
    - info: Displays the information of the directory parameters.
    """
    def __init__(self, expdatadir='', g0simdir='', simname='', simdir='', 
                 prefix='', wkdir='', species = {}):
        self.expdatadir = expdatadir
        self.g0simdir = g0simdir
        self.simname = simname
        self.simdir = g0simdir + simdir
        self.wkdir = wkdir
        self.datadir = g0simdir + simdir + simname +'/' + wkdir
        self.prefix = prefix # prefix for the data files
        self.fileprefix = self.datadir + prefix # prefix for the data files + full path
        self.data_file_dict = {}
        self.species = species
        self.data_file_dict = {}
        self.set_data_file_dict()
        self.all_field_list = []
        self.default_mom_type = None
        
    def set_data_file_dict(self):
        '''
        Sets up the data field dictionary which indicates how each 
        possible scalar field can be found.

        Parameters:
        - keys (list): List of keys for the data fields.
        - files (list): List of file names corresponding to the data fields.
        - comps (list): List of components to look for in the files.
        - BiMaxwellian (bool): Flag indicating if BiMaxwellian moments are used.
        - species (dict): Dictionary of species information.
        '''
        data_file_dict = {}
        gnames   = ['x','y','z','vpar','mu']

        # add equilibrium info
        # Magnetic field amplitude
        data_file_dict['Bmag'+'file']   = 'bmag'
        data_file_dict['Bmag'+'comp']   = 0
        data_file_dict['Bmag'+'gnames'] = gnames[0:3]

        # normalized b field
        for i_ in range(3):
            data_file_dict['b_'+gnames[i_]+'file']   = 'b_i'
            data_file_dict['b_'+gnames[i_]+'comp']   = i_
            data_file_dict['b_'+gnames[i_]+'gnames'] = gnames[0:3]

        # Jacobian
        data_file_dict['Jacobian'+'file']   = 'jacobgeo'
        data_file_dict['Jacobian'+'comp']   = 0
        data_file_dict['Jacobian'+'gnames'] = gnames[0:3]

        # metric coefficients
        counter_ = 0
        for i_ in range(3):
            iname = gnames[i_]
            for j_ in range(i_,3):
                jname = gnames[j_]
                gijname = 'g_'+iname+jname
                data_file_dict[gijname+'file']   = 'g_ij'
                data_file_dict[gijname+'comp']   = counter_
                data_file_dict[gijname+'gnames'] = gnames[0:3]
                gijname = 'g'+iname+jname
                data_file_dict[gijname+'file']   = 'gij'
                data_file_dict[gijname+'comp']   = counter_
                data_file_dict[gijname+'gnames'] = gnames[0:3]
                counter_ += 1

        # add electrostatic field info
        data_file_dict['phi'+'file'] = 'field'
        data_file_dict['phi'+'comp'] = 0
        data_file_dict['phi'+'gnames'] = gnames[0:3]
        
        for spec in self.species.values():
            s_        = spec.name
            shortname = spec.nshort
            
            for add_source in [False,True]:
                keys   = []
                prefix = []
                comps  = []
                spec = s_ + '_source' if add_source else s_
                
                # Find a file type where we can find the moment data.
                mtype = -1
                for moment_type in ['M0', 'MaxwellianMoments', 'BiMaxwellianMoments', 'HamiltonianMoments']:
                    file_name = f"-{s_}_{moment_type}_0.gkyl"
                    file_name = self.fileprefix + file_name
                    if os.path.exists(file_name):
                        mtype = moment_type
                        self.default_mom_type = mtype
                        break
                if mtype == -1:
                    print(f"No moments file found for species {s_}.")
                    continue
                # add default moments interface        
                keys  += ['n','upar','Tpar','Tperp','qpar','qperp']
                if mtype == 'M0':
                    comps  += [0,0,0,0,0,0]
                    prefix += [spec+'_M0',spec+'_M1',spec+'_M2par',spec+'_M2perp',spec+'_M3par',spec+'_M3perp']
                else:
                    comps  += [0,1,2,3,0,0]
                    prefix += 6*[spec+'_'+mtype]

                # Add Maxwellian moments
                keys   += ['MM_n','MM_upar','MM_T']
                comps  += [0,1,2]
                prefix += 3*[spec+'_MaxwellianMoments']

                # Add biMaxwellian moments
                keys   += ['BM_n','BM_upar','BM_Tpar','BM_Tperp']
                comps  += [0,1,2,3]
                prefix += 4*[spec+'_BiMaxwellianMoments']

                # add Hamiltonian moments
                keys   += ['HM_n','HM_mv','HM_H']
                comps  += [0,1,2]
                prefix += 3*[spec+'_HamiltonianMoments']
                        
                # add moments info        
                keys   += ['M0','M1','M2','M2par','M2perp','M3par','M3perp']
                comps  += [0,0,0,0,0,0,0]
                prefix += [spec+'_M0',spec+'_M1',spec+'_M2',spec+'_M2par',spec+'_M2perp',spec+'_M3par',spec+'_M3perp']
                
                # add distribution functions
                keys   += ['f']
                comps  += [0]
                prefix += [spec]
                  
                # add to the data_file_dict
                for i in range(len(keys)):
                    k = 'src_'+keys[i]+shortname if add_source else keys[i]+shortname
                    data_file_dict[k+'file'] = prefix[i]
                    data_file_dict[k+'comp'] = comps[i]
                    data_file_dict[k+'gnames'] = gnames[0:3]
        
        self.data_file_dict = data_file_dict
        
    @staticmethod
    def get_default_units_dict(species):
        """
        Returns the default units dictionary for various quantities.

        Parameters:
        - species (dict): Dictionary of species information.

        Returns:
        - dict: Default units dictionary.
        """
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
            ['Bmag', r'$B$','T'],
            ['g_xx', r'$g_{xx}$', ''],
            ['g_xy', r'$g_{xy}$', ''],
            ['g_xz', r'$g_{xz}$', ''],
            ['g_yy', r'$g_{yy}$', ''],
            ['g_yz', r'$g_{yz}$', ''],
            ['g_zz', r'$g_{zz}$', ''],
            ['gxx', r'$g^{xx}$', ''],
            ['gxy', r'$g^{xy}$', ''],
            ['gxz', r'$g^{xz}$', ''],
            ['gyy', r'$g^{yy}$', ''],
            ['gyz', r'$g^{yz}$', ''],
            ['gzz', r'$g^{zz}$', ''],
            ]
        # add routinely other quantities of interest
        for spec in species.values():
            s_ = spec.nshort
            for add_source in [False, True]:
                src_ = 'src_' if add_source else ''
                S_ = 'S' if add_source else ''
                # distribution functions
                default_qttes.append(['%sf%s'%(src_,s_), r'%s$f_%s$'%(S_,s_), '[f]'])
                # Moments
                default_qttes.append(['%sM0%s'%(src_,s_), r'%s$M_{0%s}$'%(S_,s_), r'm$^{-3}$'])
                default_qttes.append(['%sM1%s'%(src_,s_), r'%s$M_{1%s}$'%(S_,s_), r'm$^{-2}$/s'])
                default_qttes.append(['%sM2%s'%(src_,s_), r'%s$M_{2%s}$'%(S_,s_), r'J/kg/m$^{3}$'])
                default_qttes.append(['%sM2par%s'%(src_,s_), r'%s$M_{2\parallel %s}$'%(S_,s_), r'J/kg/m$^{3}$'])
                default_qttes.append(['%sM2perp%s'%(src_,s_), r'%s$M_{2\perp %s}$'%(S_,s_), r'J/kg/m$^{3}$'])
                default_qttes.append(['%sM3par%s'%(src_,s_), r'%s$M_{3\parallel %s}$'%(S_,s_), r'J/kg/m$^{2}/s$'])
                default_qttes.append(['%sM3perp%s'%(src_,s_), r'%s$M_{3\perp %s}$'%(S_,s_), r'J/kg/m$^{2}/s$'])
                # Generic moments
                default_qttes.append(['%sn%s'%(src_,s_), r'%s$n_%s$'%(S_,s_), r'm$^{-3}$'])
                default_qttes.append(['%supar%s'%(src_,s_), r'%s$u_{\parallel %s}$'%(S_,s_), 'm/s'])
                default_qttes.append(['%sTpar%s'%(src_,s_), r'%s$T_{\parallel %s}$'%(S_,s_), 'J/kg'])
                default_qttes.append(['%sTperp%s'%(src_,s_), r'%s$T_{\perp %s}$'%(S_,s_), 'J/kg'])
                # Maxwellian moments
                default_qttes.append(['%sMM_n%s'%(src_,s_), r'%s$n_%s$'%(S_,s_), r'm$^{-3}$'])
                default_qttes.append(['%sMM_upar%s'%(src_,s_), r'%s$u_{\parallel %s}$'%(S_,s_), 'm/s'])
                default_qttes.append(['%sMM_T%s'%(src_,s_), r'%s$T_{\parallel %s}$'%(S_,s_), 'J/kg'])
                # BiMaxwellian moments
                default_qttes.append(['%sBM_n%s'%(src_,s_), r'%s$n_%s$'%(S_,s_), r'm$^{-3}$'])
                default_qttes.append(['%sBM_upar%s'%(src_,s_), r'%s$u_{\parallel %s}$'%(S_,s_), 'm/s'])
                default_qttes.append(['%sBM_Tpar%s'%(src_,s_), r'%s$T_{\parallel %s}$'%(S_,s_), 'J/kg'])
                default_qttes.append(['%sBM_Tperp%s'%(src_,s_), r'%s$T_{\perp %s}$'%(S_,s_), 'J/kg'])
                # Hamiltonian moments
                default_qttes.append(['%sHM_n%s'%(src_,s_), r'%s$n_%s$'%(S_,s_), r'm$^{-3}$'])            
                default_qttes.append(['%sHM_mv%s'%(src_,s_), r'%s$p_%s$'%(S_,s_), r'kg m/s m$^{-3}$'])            
                default_qttes.append(['%sHM_H%s'%(src_,s_), r'%s$H_%s$'%(S_,s_), r'J m$^{-3}$'])            
            
        #-The above defined fields are all simple quantities in the sense that 
        # composition=[identification] and so receipe = composition[0]
        def identity(gdata_list):
            return pgkyl_.get_values(gdata_list[0])
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
                b_j     = pgkyl_.get_values(gdata_list[0])
                b_k     = pgkyl_.get_values(gdata_list[1])
                Bmag    = pgkyl_.get_values(gdata_list[2])
                Jacob   = pgkyl_.get_values(gdata_list[3])
                grids   = gdata_list[0].get_grid()
                j       = np.mod(i+1,3)
                k       = np.mod(i+2,3)
                jgrid   = grids[j][:-1]
                kgrid   = grids[k][:-1]
                phi     = pgkyl_.get_values(gdata_list[4])
                dphidj  = np.gradient(phi, jgrid, axis=j)
                dphidk  = np.gradient(phi, kgrid, axis=k)
                return -(dphidj*b_k - dphidk*b_j)/Jacob/Bmag
            default_qttes.append([name,symbol,units,field2load,receipe_vExB])
            for j_ in range(len(directions)):
                cj_ = directions[j_] # direction of the flux component
                # ExB shearing rate
                name       = 'sExB%s_%s'%(ci_,cj_)
                symbol     = r'$\partial_%s v_{E,%s}$'%(cj_,ci_)
                units      = r'1/s'
                field2load = ['b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian','phi']
                # The receipe depends on the direction 
                # because of the phi derivative
                def receipe_sExB(gdata_list,i=i_,j=j_):
                    vExBi = receipe_vExB(gdata_list,i=i)
                    grids = gdata_list[0].get_grid()
                    jgrid = grids[j][:-1]
                    sExB  = np.gradient(vExBi, jgrid, axis=j)
                    return sExB
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
                upar = pgkyl_.get_values(gdata_list[0])
                Tom  = (pgkyl_.get_values(gdata_list[1]) + 2.0*pgkyl_.get_values(gdata_list[2]))/3.0
                vt   = np.sqrt(2*Tom)
                return upar/vt
            default_qttes.append([name,symbol,units,field2load,receipe_spars])      

            #total temperature
            name       = 'T%s'%(s_)
            symbol     = r'$T_{%s}$'%(s_)
            units      = 'J/kg' # T is stored as T/m in gkeyll
            field2load = ['Tpar%s'%(s_),'Tperp%s'%(s_)]
            def receipe_Ttots(gdata_list):
                return (pgkyl_.get_values(gdata_list[0]) + 2.0*pgkyl_.get_values(gdata_list[1]))/3.0
            default_qttes.append([name,symbol,units,field2load,receipe_Ttots])

            #Kinetic energy density speciewise: Wkins = int dv3 1/2 ms vpar^2 + mus B
            name       = 'Wkin%s'%(s_)
            symbol     = r'$W_{k,%s}$'%(s_) 
            units      = r'J/m$^3$'
            field2load = ['n%s'%s_,'Tpar%s'%(s_),'Tperp%s'%(s_)]
            def receipe_Wkins(gdata_list,m=spec.m):
                dens = pgkyl_.get_values(gdata_list[0])
                Ttot = receipe_Ttots(gdata_list[1:3])
                return dens*m*Ttot
            default_qttes.append([name,symbol,units,field2load,receipe_Wkins])

            #Fluid kinetic energy density speciewise: Wflu = 1/2 m*n*upar^2
            name       = 'Wflu%s'%(s_)
            symbol     = r'$W_{f,%s}$'%(s_) 
            units      = r'J/m$^3$'
            field2load = ['n%s'%s_,'upar%s'%(s_)]
            def receipe_Wflus(gdata_list,m=spec.m):
                dens = pgkyl_.get_values(gdata_list[0])
                upar = pgkyl_.get_values(gdata_list[1])
                return dens*m*np.power(upar,2)/2.0
            default_qttes.append([name,symbol,units,field2load,receipe_Wflus])

            #Potential energy density speciewise: Wpots = qs phi
            name       = 'Wpot%s'%(s_)
            symbol     = r'$W_{p,%s}$'%(s_) 
            units      = r'J/m$^3$'
            field2load = ['phi','n%s'%s_]
            def receipe_Wpots(gdata_list,q=spec.q):
                qphi = q*pgkyl_.get_values(gdata_list[0])
                dens = pgkyl_.get_values(gdata_list[1])
                return dens*qphi
            default_qttes.append([name,symbol,units,field2load,receipe_Wpots])

            #total energy density speciewise: Ws = Wkins + Wflu + Wpots
            name       = 'Wtot%s'%(s_)
            symbol     = r'$W_{%s}$'%(s_) 
            units      = r'J/m$^3$'
            field2load = ['phi','n%s'%s_,'upar%s'%(s_),'Tpar%s'%(s_),'Tperp%s'%(s_)]
            # We need the mass and the charge to converte J/kg (temp) and V (phi) in Joules
            def receipe_Ws(gdata_list,q=spec.q,m=spec.m):
                qphi = q*pgkyl_.get_values(gdata_list[0])
                dens = pgkyl_.get_values(gdata_list[1])
                upar = pgkyl_.get_values(gdata_list[2])
                Ttot = receipe_Ttots(gdata_list[3:5])
                return dens*(m*np.power(upar,2)/2 + m*Ttot + qphi)
            default_qttes.append([name,symbol,units,field2load,receipe_Ws])

            #kinetic energy (M2 moment) speciewise: Wkin = 1/2 m*M2
            name       = 'WkinM2%s'%(s_)
            symbol     = r'$W_{k,M2,%s}$'%(s_)
            units      = r'J/m$^3$'
            field2load = ['M2%s'%s_]
            def receipe_WkinM2s(gdata_list,m=spec.m):
                return m*pgkyl_.get_values(gdata_list[0])/2.0
            default_qttes.append([name,symbol,units,field2load,receipe_WkinM2s])

            #total pressure speciewise
            name = 'p%s'%(s_)
            symbol = r'$p_{%s}$'%(s_)
            units = 'J/kg/m$^{3}$'
            field2load = ['n%s'%(s_),'Tpar%s'%(s_),'Tperp%s'%(s_)]
            def receipe_ptots(gdata_list):
                Ttot = receipe_Ttots(gdata_list[1:3])
                return pgkyl_.get_values(gdata_list[0])*Ttot
            default_qttes.append([name,symbol,units,field2load,receipe_ptots])

            #parallel pressures
            name       = 'ppar%s'%(s_)
            symbol     = r'$p_{\parallel,%s}$'%(s_)
            units      = 'J/kg/m$^{3}$'
            field2load = ['n%s'%(s_),'Tpar%s'%(s_)]
            def receipe_ppars(gdata_list):
                return pgkyl_.get_values(gdata_list[0])*pgkyl_.get_values(gdata_list[1])/3.0
            default_qttes.append([name,symbol,units,field2load,receipe_ppars])

            #perpendicular pressures
            name       = 'pperp%s'%(s_)
            symbol     = r'$p_{\perp,%s}$'%(s_)
            units      = 'J/kg/m$^{3}$'
            field2load = ['n%s'%(s_),'Tperp%s'%(s_)]
            def receipe_pperps(gdata_list):
                return pgkyl_.get_values(gdata_list[0])*pgkyl_.get_values(gdata_list[1])*2.0/3.0     
            default_qttes.append([name,symbol,units,field2load,receipe_pperps])

            #normalized pressure beta
            name = 'beta%s'%(s_)
            symbol = r'$\beta_{%s}$'%(s_)
            units = r'$\%$'
            field2load = ['n%s'%(s_),'Tpar%s'%(s_),'Tperp%s'%(s_),'Bmag']
            def receipe_betas(gdata_list,m=spec.m):
                mu0 = 4.0*np.pi*1e-7
                dens = pgkyl_.get_values(gdata_list[0])
                Ttot = receipe_Ttots(gdata_list[1:3])
                Bmag = pgkyl_.get_values(gdata_list[3])
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
                    b_j     = pgkyl_.get_values(gdata_list[0])
                    b_k     = pgkyl_.get_values(gdata_list[1])
                    Bmag    = pgkyl_.get_values(gdata_list[2])
                    Jacob   = pgkyl_.get_values(gdata_list[3])
                    Tperp   = pgkyl_.get_values(gdata_list[4])*m
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
                    density = pgkyl_.get_values(gdata_list[5])
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
                    density = pgkyl_.get_values(gdata_list[5])
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
                    density = pgkyl_.get_values(gdata_list[5])
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
                    density = pgkyl_.get_values(gdata_list[5])
                    Ttot    = receipe_Ttots(gdata_list[6:8])
                    #T is converted to joules by mass
                    return density * m*Ttot * vgB
                default_qttes.append([name,symbol,units,field2load,receipe_gradB_hflux_s])

        # Species independent quantities

        # total M2 kinetic energy density
        name       = 'WkinM2'
        symbol     = r'$W_{k,M2}$'
        units      = r'J/m$^3$'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('M2%s'%s_)
        def receipe_WkinM2(gdata_list,species=species):
            fout = 0.0
            for spec in species.values():
                fout += receipe_WkinM2s(gdata_list, m=spec.m)
            return fout
        default_qttes.append([name,symbol,units,field2load,receipe_WkinM2])
        
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
                fout += spec.q*pgkyl_.get_values(gdata_list[0+k])
                k    += 1
            return fout
        default_qttes.append([name,symbol,units,field2load,receipe_qdens])

        #ion-electron temperature ratio
        name       = 'Tratio'
        symbol     = r'$T_{i}/T_e$'
        units      = '' # T is stored as T/m in gkeyll
        field2load = ['Tpare','Tperpe','Tpari','Tperpi']
        def receipe_Tratio(gdata_list,species=species):
            Te = (pgkyl_.get_values(gdata_list[0]) + 2.0*pgkyl_.get_values(gdata_list[1]))/3.0
            Ti = (pgkyl_.get_values(gdata_list[2]) + 2.0*pgkyl_.get_values(gdata_list[3]))/3.0
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

        #thermal energy : \sum_s W_kins = \sum_s int dv3 1/2 ms vpar^2 + mus B
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
                phi     = pgkyl_.get_values(gdata_list[0])
                grids   = gdata_list[0].get_grid()
                grid    = grids[i][:-1]
                return -np.gradient(phi, grid, axis=i)
            default_qttes.append([name,symbol,units,field2load,receipe_Ei])

        #source power
        name       = 'pow_src'
        symbol     = r'$P_{src}$'
        units      = r'W/m$^3$'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('n_src%s'%s_)
            field2load.append('Tpar_src%s'%(s_))
            field2load.append('Tperp_src%s'%(s_))
        def receipe_src_pow(gdata_list,species=species):
            fout = 0.0
            k    = 0
            for spec in species.values():
                fout += receipe_Wkins(gdata_list[0+k:3+k],m=spec.m)
                k += 3
            return fout 
        default_qttes.append([name,symbol,units,field2load,receipe_src_pow])

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
            ck_ = directions[np.mod(i+2,3)] # direction coord + 2
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

        # now that all fields where declared, we create a list of all the available fields
        all_field_list = []
        for key in default_units_dict:
            if key.endswith('symbol'):
                all_field_list.append(key[:-6])
        # eliminate duplicates
        all_field_list = list(set(all_field_list))
        
        # add default colormap for each fields
        positive_fields = ['Bmag','pow_src'] # spec. indep
        
        spec_dep_fields = ['M0','M2','M2par','M2perp', 'M3par','M3perp',
                           'n','T','Tpar','Tperp','p',
                           'BM_n','BM_Tpar','BM_Tperp',
                           'src_M0','src_M2','src_M2par','src_M2perp', 'src_M3par','src_M3perp',
                           'src_n','src_T','src_Tpar','src_Tperp','src_p',
                           'src_BM_n','src_BM_Tpar','src_BM_Tperp',
                           'f','src_f']
        for sdepfield in spec_dep_fields:
            for spec in species.values():
                positive_fields.append(sdepfield+spec.nshort)
        
        for field in all_field_list:
            if field in positive_fields:
                default_units_dict[field+'colormap'] = 'inferno'
            else:
                default_units_dict[field+'colormap'] = 'bwr'

        # Return a copy of the units dictionary
        return copy.copy(default_units_dict)

    def info(self):
        default_dict = DataParam.get_default_units_dict(self.species)
        # Create a table to display the data
        print(f"A table of the default quantities and their default units:")
        print(f"| {'Quantity':<15} | {'Symbol':<30} | {'Units':<20} |")
        print(f"|{'-' * 17}|{'-' * 32}|{'-' * 22}|")

        for key in default_dict:
            if key.endswith('symbol'):  # Check for a specific type of key
                quantity = key[:-6]  # Remove the suffix to get the base name
                if not quantity in ['x','y','z','ky','wavelen','vpar','mu','t','fi']:
                    symbol = default_dict[f'{quantity}symbol']
                    units = default_dict.get(f'{quantity}units', 'N/A')
                    # Format as a table row
                    print(f"| {quantity:<15} | {symbol:<30} | {units:<20} |")