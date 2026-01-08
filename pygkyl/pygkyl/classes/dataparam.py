import copy
import numpy as np
import os
from ..interfaces import pgkyl_interface as pgkyl_
from ..utils import file_utils as file_utils
from ..tools import phys_tools
import glob

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
                 prefix='', wkdir='', species = {}, checkfiles=True):
        self.expdatadir = expdatadir
        self.g0simdir = g0simdir
        self.simname = simname
        self.simdir = g0simdir + simdir
        self.wkdir = wkdir
        self.datadir = g0simdir + simdir + simname +'/' + wkdir
        self.prefix = prefix # prefix for the data files
        self.fileprefix = self.datadir + prefix # prefix for the data files + full path
        self.species = species
        self.file_info_dict = {}
        self.set_data_file_dict(checkfiles=checkfiles)
        self.default_mom_type = None
        self.field_info_dict = self.get_default_units_dict(species) # dictionary of the default parameters for all fields
        self.time_independent_fields = [
            'b_x', 'b_y', 'b_z', 'Jacobian', 'Bmag', 
            'g_xx', 'g_xy', 'g_xz', 'g_yy', 'g_yz', 'g_zz',
            'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
        
    def set_data_file_dict(self, checkfiles=True):
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
        file_dict = {}
        gnames   = ['x','y','z','vpar','mu']

        # add equilibrium info
        # Magnetic field amplitude
        file_dict['Bmag'+'file']   = 'bmag'
        file_dict['Bmag'+'comp']   = 0
        file_dict['Bmag'+'gnames'] = gnames[0:3]

        # normalized b field
        for i_ in range(3):
            file_dict['b_'+gnames[i_]+'file']   = 'b_i'
            file_dict['b_'+gnames[i_]+'comp']   = i_
            file_dict['b_'+gnames[i_]+'gnames'] = gnames[0:3]

        # Jacobian
        file_dict['Jacobian'+'file']   = 'jacobgeo'
        file_dict['Jacobian'+'comp']   = 0
        file_dict['Jacobian'+'gnames'] = gnames[0:3]

        # metric coefficients
        counter_ = 0
        for i_ in range(3):
            iname = gnames[i_]
            for j_ in range(i_,3):
                jname = gnames[j_]
                gijname = 'g_'+iname+jname
                file_dict[gijname+'file']   = 'g_ij'
                file_dict[gijname+'comp']   = counter_
                file_dict[gijname+'gnames'] = gnames[0:3]
                gijname = 'g'+iname+jname
                file_dict[gijname+'file']   = 'gij'
                file_dict[gijname+'comp']   = counter_
                file_dict[gijname+'gnames'] = gnames[0:3]
                counter_ += 1

        # add electrostatic field info
        file_dict['phi'+'file'] = 'field'
        file_dict['phi'+'comp'] = 0
        file_dict['phi'+'gnames'] = gnames[0:3]
        
        # add parallel component of the vector potential
        file_dict['Apar'+'file'] = 'apar'
        file_dict['Apar'+'comp'] = 0
        file_dict['Apar'+'gnames'] = gnames[0:3]
        # and its time derivative
        file_dict['Apardot'+'file'] = 'apardot'
        file_dict['Apardot'+'comp'] = 0
        file_dict['Apardot'+'gnames'] = gnames[0:3]

        # flan interface
        file_dict['flan'+'file'] = 'flan'
        file_dict['flan'+'comp'] = 0
        file_dict['flan'+'gnames'] = gnames[0:3]   
             
        for spec in self.species.values():
            s_        = spec.name
            shortname = spec.nshort
            
            for add_source in [False,True]:
                keys   = []
                prefix = []
                comps  = []
                spec = s_ + '_source' if add_source else s_
                
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
                
                # add three moments
                keys   += ['3M_M0','3M_M1','3M_M2']
                comps  += [0,1,2]
                prefix += 3*[spec+'_M0M1M2']
                        
                # add moments info        
                keys   += ['M0','M1','M2','M2par','M2perp','M3par','M3perp']
                comps  += [0,0,0,0,0,0,0]
                prefix += [spec+'_M0',spec+'_M1',spec+'_M2',spec+'_M2par',spec+'_M2perp',spec+'_M3par',spec+'_M3perp']
                
                # add default moments interface        
                # Find a file type where we can find the moment data.
                if checkfiles:
                    mtype = 'none'
                    for moment_type in ['BiMaxwellianMoments', 'HamiltonianMoments', 'M0', 'M0M1M2']:
                        pattern = f"{self.fileprefix}-{spec}_{moment_type}_*.gkyl"
                        files = glob.glob(pattern)
                        if files:
                            file_name = self.simdir + os.path.basename(files[0])
                        else:
                            file_name = self.fileprefix + f"-{spec}_{moment_type}_0.gkyl"
                        if os.path.exists(file_name):
                            mtype = moment_type
                            self.default_mom_type = mtype
                            break
                    if mtype == 'none':
                        # print(f"No moments file found for species {spec}. (recall, we do not support Maxwellian moments yet)")
                        # print(f"Check the file name pattern: {self.fileprefix}-{spec}_{moment_type}_*.gkyl")
                        continue
                else:
                    mtype = 'BiMaxwellianMoments'
                    self.default_mom_type = mtype
                    
                keys  += ['n','upar','Tpar','Tperp','qpar','qperp']
                if self.default_mom_type == 'M0':
                    comps  += [0,0,0,0,0,0]
                    prefix += [spec+'_M0',spec+'_M1',spec+'_M2par',spec+'_M2perp',spec+'_M3par',spec+'_M3perp']
                elif self.default_mom_type == 'BiMaxwellianMoments':
                    comps  += [0,1,2,3,0,0]
                    prefix += 6*[spec+'_'+mtype]
                elif self.default_mom_type == 'HamiltonianMoments':
                    comps  += [0,1,2,0,0,0]
                    prefix += 6*[spec+'_'+mtype]
                elif self.default_mom_type == 'M0M1M2':
                    comps  += [0,1,2,0,0,0]
                    prefix += 6*[spec+'_'+mtype]

                # add distribution functions
                keys   += ['f']
                comps  += [0]
                prefix += [spec]
                
                for i in range(len(keys)):
                    k = 'src_'+keys[i]+shortname if add_source else keys[i]+shortname
                    file_dict[k+'file'] = prefix[i]
                    file_dict[k+'comp'] = comps[i]
                    if keys[i] == 'f':
                        file_dict[k+'gnames'] = gnames
                    else:
                        file_dict[k+'gnames'] = gnames[0:3]
                    
        # Store a list of all the different file names we may look for.
        file_dict['names'] = []
        for key in file_dict.keys():
            if 'file' in key:
                file_dict['names'].append(file_dict[key])
        # Remove duplicates from the file list
        file_dict['names'] = list(set(file_dict['names']))
        
        # Sort the file keys alphabetically
        file_dict = dict(sorted(file_dict.items()))
        
        self.file_info_dict = file_dict
        
    @staticmethod
    def get_available_frames(simulation):
        """
        This function builds a list of all available frames per key in the file_info_dict.
        """
        available_frames = {}
        file_dict = simulation.data_param.file_info_dict
        filelist = file_dict['names']
        for file in filelist:
            available_frames[file] = file_utils.find_available_frames(simulation, file)
        return available_frames
            
    @staticmethod
    def get_default_units_dict(species):
        """
        This builds all the default units for the fields that we are able to plot.

        Parameters:
        - species (dict): Dictionary of species information.

        Returns:
        - dict: units, symbols, colormap, composition, and recipe for each field.
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
            ['Apar', r'$A_\parallel$', 'V s/m'],
            ['Apardot', r'$\partial_t A_\parallel$', 'V/m'],
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
            default_qttes.append(['vpar%s'%s_, r'$v_{\parallel %s}$'%s_, 'm/s'])
            default_qttes.append(['mu%s'%s_, r'$\mu_%s$'%s_, 'J/T'])
            for add_source in [False, True]:
                src_ = 'src_' if add_source else ''
                S_ = 'S' if add_source else ''
                # distribution functions
                default_qttes.append(['%sf%s'%(src_,s_), r'%s$f_%s$'%(S_,s_), '[f]'])
                # Moments (id: src_xs or xs)
                default_qttes.append(['%sM0%s'%(src_,s_), r'%s$M_{0%s}$'%(S_,s_), r'm$^{-3}$'])
                default_qttes.append(['%sM1%s'%(src_,s_), r'%s$M_{1%s}$'%(S_,s_), r'm$^{-2}$/s'])
                default_qttes.append(['%sM2%s'%(src_,s_), r'%s$M_{2%s}$'%(S_,s_), r'J/kg/m$^{3}$'])
                default_qttes.append(['%sM2par%s'%(src_,s_), r'%s$M_{2\parallel %s}$'%(S_,s_), r'J/kg/m$^{3}$'])
                default_qttes.append(['%sM2perp%s'%(src_,s_), r'%s$M_{2\perp %s}$'%(S_,s_), r'J/kg/m$^{3}$'])
                default_qttes.append(['%sM3par%s'%(src_,s_), r'%s$M_{3\parallel %s}$'%(S_,s_), r'J/kg/m$^{2}/s$'])
                default_qttes.append(['%sM3perp%s'%(src_,s_), r'%s$M_{3\perp %s}$'%(S_,s_), r'J/kg/m$^{2}/s$'])
                # Generic moments (id: src_xs or xs)
                default_qttes.append(['%sn%s'%(src_,s_), r'%s$n_%s$'%(S_,s_), r'm$^{-3}$'])
                default_qttes.append(['%supar%s'%(src_,s_), r'%s$u_{\parallel %s}$'%(S_,s_), 'm/s'])
                default_qttes.append(['%sTpar%s'%(src_,s_), r'%s$T_{\parallel %s}$'%(S_,s_), 'J/kg'])
                default_qttes.append(['%sTperp%s'%(src_,s_), r'%s$T_{\perp %s}$'%(S_,s_), 'J/kg'])
                # Maxwellian moments (id: src_MM_xs or MM_xs)
                default_qttes.append(['%sMM_n%s'%(src_,s_), r'%s$n_%s$'%(S_,s_), r'm$^{-3}$'])
                default_qttes.append(['%sMM_upar%s'%(src_,s_), r'%s$u_{\parallel %s}$'%(S_,s_), 'm/s'])
                default_qttes.append(['%sMM_T%s'%(src_,s_), r'%s$T_{\parallel %s}$'%(S_,s_), 'J/kg'])
                # BiMaxwellian moments (id: src_BM_xs or BM_xs)
                default_qttes.append(['%sBM_n%s'%(src_,s_), r'%s$n_%s$'%(S_,s_), r'm$^{-3}$'])
                default_qttes.append(['%sBM_upar%s'%(src_,s_), r'%s$u_{\parallel %s}$'%(S_,s_), 'm/s'])
                default_qttes.append(['%sBM_Tpar%s'%(src_,s_), r'%s$T_{\parallel %s}$'%(S_,s_), 'J/kg'])
                default_qttes.append(['%sBM_Tperp%s'%(src_,s_), r'%s$T_{\perp %s}$'%(S_,s_), 'J/kg'])
                # Hamiltonian moments (id: src_HM_xs or HM_xs)
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
            
            # curl b
            name       = 'curlb_%s'%(ci_)
            symbol     = r'$(\nabla \times b)_{%s}$'%(ci_)
            units      = r'1/m'
            field2load = ['b_%s'%cj_,'b_%s'%ck_,'Jacobian']
            def receipe_curl_b(gdata_list,i=i_):
                j = np.mod(i+1,3)
                k = np.mod(i+2,3)
                bj     = pgkyl_.get_values(gdata_list[0])
                bk     = pgkyl_.get_values(gdata_list[1])
                Jacob   = pgkyl_.get_values(gdata_list[2])
                grids   = pgkyl_.get_grid(gdata_list[0])
                jgrid   = grids[j][:-1]
                kgrid   = grids[k][:-1]
                dbjdk = np.gradient(bj, kgrid, axis=k)
                dbkdj = np.gradient(bk, jgrid, axis=j)
                return (dbjdk - dbkdj)/Jacob
            default_qttes.append([name,symbol,units,field2load,receipe_curl_b])

            # Curvature (- b x ( curl b)
            name       = 'curv_%s'%(ci_)
            symbol     = r'$\kappa_{%s}$'%(ci_)
            units      = r'm$^{-2}$'
            field2load = ['b_%s'%ci_,'b_%s'%cj_,'b_%s'%ck_,'Jacobian']
            def receipe_curv(gdata_list,i=i_):
                j = np.mod(i+1,3)
                k = np.mod(i+2,3)
                bj     = pgkyl_.get_values(gdata_list[1])
                bk     = pgkyl_.get_values(gdata_list[2])
                curlb_j = receipe_curl_b([gdata_list[j],gdata_list[k],gdata_list[-1]],i=j)
                curlb_k = receipe_curl_b([gdata_list[k],gdata_list[i],gdata_list[-1]],i=k)
                return -(bj*curlb_k - bk*curlb_j)
            default_qttes.append([name,symbol,units,field2load,receipe_curv])
            
            # ExB velocity
            name       = 'ExB_v_%s'%(ci_)
            symbol     = r'$u_{E,%s}$'%(ci_)
            units      = r'm/s'
            field2load = ['phi','b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian']
            # The receipe depends on the direction 
            # because of the phi derivative
            def receipe_vExB(gdata_list,i=i_):
                phi     = pgkyl_.get_values(gdata_list[0])
                b_j     = pgkyl_.get_values(gdata_list[1])
                b_k     = pgkyl_.get_values(gdata_list[2])
                Bmag    = pgkyl_.get_values(gdata_list[3])
                Jacob   = pgkyl_.get_values(gdata_list[4])
                grids   = pgkyl_.get_grid(gdata_list[0])
                j       = np.mod(i+1,3)
                k       = np.mod(i+2,3)
                jgrid   = grids[j][:-1]
                kgrid   = grids[k][:-1]
                dphidj  = np.gradient(phi, jgrid, axis=j)
                dphidk  = np.gradient(phi, kgrid, axis=k)
                return -(dphidj*b_k - dphidk*b_j)/Jacob/Bmag
            default_qttes.append([name,symbol,units,field2load,receipe_vExB])
            
            # Perpendicular magnetic field perturbation $\delta B_\perp = \nabla \times (A_\parallel b)$
            # following curl(Apar * b) = Apar * curl(b) + grad(Apar) x b
            name = 'dB_perp_%s'%(ci_)
            symbol = r'$\delta B_{\perp,%s}$'%(ci_)
            units = r'T'
            field2load = ['Apar','b_%s'%cj_,'b_%s'%ck_,'Jacobian']
            def receipe_dB_perp(gdata_list,i=i_):
                j = np.mod(i+1,3)
                k = np.mod(i+2,3)
                Apar   = pgkyl_.get_values(gdata_list[0])
                bj     = pgkyl_.get_values(gdata_list[1])
                bk     = pgkyl_.get_values(gdata_list[2])
                Jacob   = pgkyl_.get_values(gdata_list[3])
                grids   = pgkyl_.get_grid(gdata_list[0])
                jgrid   = grids[j][:-1]
                kgrid   = grids[k][:-1]
                result = 0.0
                # add Apar * curl(b)
                curlb = receipe_curl_b([gdata_list[1],gdata_list[2],gdata_list[3]],i=i_)
                result += Apar * curlb
                # add grad(Apar) x b
                dApardj  = np.gradient(Apar, jgrid, axis=j)
                dApardk  = np.gradient(Apar, kgrid, axis=k)
                result += (dApardj*bk - dApardk*bj)/Jacob
                return result
            default_qttes.append([name,symbol,units,field2load,receipe_dB_perp])
            
        for i_ in range(len(directions)):
            for j_ in range(len(directions)):
                ci_ = directions[i_] # direction of the derivative of vExB
                cj_ = directions[j_] # direction of vExB
                cl_ = directions[np.mod(j_+1,3)] # direction coord + 2
                ck_ = directions[np.mod(j_+2,3)] # direction coord + 2
                # ExB shearing rate
                name       = 'ExB_s_%s_%s'%(ci_,cj_)
                symbol     = r'$\partial_%s v_{E,%s}$'%(ci_,cj_)
                units      = r'1/s'
                field2load = ['phi','b_%s'%cl_,'b_%s'%ck_,'Bmag','Jacobian']
                # The receipe depends on the direction 
                # because of the phi derivative
                def receipe_sExB(gdata_list,i=i_,j=j_):
                    vExBj = receipe_vExB(gdata_list,i=j)
                    grids = pgkyl_.get_grid(gdata_list[0])
                    igrid = grids[i][:-1]
                    sExB  = np.gradient(vExBj, igrid, axis=i)
                    return sExB
                default_qttes.append([name,symbol,units,field2load,receipe_sExB])
                
                # normalized ExB shearing rate
                name       = 'norm_ExB_s_%s_%s'%(ci_,cj_)
                symbol     = r'$\partial_%s v_{E,%s}/c_s$'%(ci_,cj_)
                units      = r'1/m'
                field2load = ['phi','b_%s'%cl_,'b_%s'%ck_,'Bmag','Jacobian','Tpare','Tperpe']
                def receipe_normsExB(gdata_list,i=i_,j=j_):
                    sExB = receipe_sExB(gdata_list[:-2],i,j)
                    Te   = (pgkyl_.get_values(gdata_list[-2]) + 2.0*pgkyl_.get_values(gdata_list[-1]))/3.0
                    Te   *= species['elc'].m # convert from J/kg to J
                    # get the ion species
                    for spec in species.values():
                        if spec.nshort == 'i':
                            mi = spec.m
                    cs   = np.sqrt(Te/mi)
                    return sExB/cs
                default_qttes.append([name,symbol,units,field2load,receipe_normsExB])

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
            
            #normalized radial density gradient
            name       = 'gradlogn%s'%(s_)
            symbol     = r'$-\nabla \ln n_{%s}$'%(s_)
            units      = '1/m'
            field2load = ['n%s'%(s_)]
            def receipe_gradn(gdata_list):
                dens = pgkyl_.get_values(gdata_list[0])
                dens[dens <= 0] = np.nan # avoid log(0)
                grids = pgkyl_.get_grid(gdata_list[0])
                return -np.gradient(np.log(dens), grids[0][:-1], axis=0)
            default_qttes.append([name,symbol,units,field2load,receipe_gradn])

            #normalized radial temperature gradient
            name       = 'gradlogT%s'%(s_)
            symbol     = r'-\nabla \ln T_{%s}$'%(s_)
            units      = '1/m'
            field2load = ['Tpar%s'%(s_),'Tperp%s'%(s_)]
            def receipe_gradT(gdata_list):
                temp = pgkyl_.get_values(gdata_list[0]) + 2.0*pgkyl_.get_values(gdata_list[1])/3.0
                temp[temp <= 0] = np.nan # avoid log(0)
                grids = pgkyl_.get_grid(gdata_list[0])
                return -np.gradient(np.log(temp), grids[0][:-1], axis=0)
            default_qttes.append([name,symbol,units,field2load,receipe_gradT])

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
            
            #source density
            name = 'src_n%s'%(s_)
            symbol = r'$n_{S,%s}$'%(s_)
            units = r'm$^{-3}/s$'
            field2load = ['src_n%s'%(s_)]
            def receipe_src_ns(gdata_list):
                """
                Source density: n = n_s
                """
                return pgkyl_.get_values(gdata_list[0])
            default_qttes.append([name,symbol,units,field2load,receipe_src_ns])
            
            #source power
            name = 'src_P%s'%(s_)
            symbol = r'$P_{S,%s}$'%(s_)
            units = r'W/m$^{3}$'
            field2load = ['src_HM_H%s'%(s_),'phi','src_n%s'%(s_)]
            def receipe_src_Ps(gdata_list,q=spec.q):
                """
                Source power density: P = d(H - q n phi)/dt
                """
                Hdot = pgkyl_.get_values(gdata_list[0])
                phi = pgkyl_.get_values(gdata_list[1])
                ndot = pgkyl_.get_values(gdata_list[2])
                return Hdot - ndot * q * phi
            default_qttes.append([name,symbol,units,field2load,receipe_src_Ps])
            
            #source temperature
            name = 'src_T%s'%(s_)
            symbol = r'$T_{S,%s}$'%(s_)
            units = r'eV'
            field2load = ['src_HM_H%s'%(s_),'phi','src_n%s'%(s_)]
            def receipe_src_Ts(gdata_list,q=spec.q):
                """
                Source temperature density: T = 2/3 Edot / ndot
                """
                Edot = receipe_src_Ps(gdata_list,q=q)
                ndot = pgkyl_.get_values(gdata_list[2])
                return 2/3 * Edot / ndot
            default_qttes.append([name,symbol,units,field2load,receipe_src_Ts])

            # Larmor radius
            name = 'rho%s'%(s_)
            symbol = r'$\rho_{%s}$'%(s_)
            units = r'm'
            field2load = ['Tperp%s'%(s_),'Bmag']
            def receipe_rhos(gdata_list,q=spec.q,m=spec.m):
                """
                Larmor radius: rho = sqrt(m Tperp) / (|q| B)
                """
                Tperp = pgkyl_.get_values(gdata_list[0]) * phys_tools.kB
                Bmag = pgkyl_.get_values(gdata_list[1])
                # remove unphysical values
                Tperp[Tperp <= 0] = np.nan # avoid sqrt(negative
                Bmag[Bmag <= 0] = np.nan # avoid div by 0
                return np.sqrt(m * Tperp) / (np.abs(q) * Bmag)
            default_qttes.append([name,symbol,units,field2load,receipe_rhos])
            
            for rpec in species.values():
                r_ = rpec.nshort
                
                # Collision frequency
                name = 'nu%s%s'%(s_,r_)
                symbol = r'$\nu_{%s%s}$'%(s_,r_)
                units = r'1/s'
                field2load = ['n%s'%(s_), 'Tpar%s'%(s_), 'Tperp%s'%(s_),
                              'n%s'%(r_), 'Tpar%s'%(r_), 'Tperp%s'%(r_),
                              'Bmag']
                def receipe_nu(gdata_list,qs=spec.q,ms=spec.m,qr=rpec.q,mr=rpec.m):
                    ns = pgkyl_.get_values(gdata_list[0])
                    Ts = receipe_Ttots(gdata_list[1:3])*ms
                    nr = pgkyl_.get_values(gdata_list[3])
                    Tr = receipe_Ttots(gdata_list[4:6])*mr
                    Bmag = pgkyl_.get_values(gdata_list[6])
                    return phys_tools.collision_freq(ns, qs, ms, Ts, nr, qr, mr, Tr, Bmag)
                default_qttes.append([name,symbol,units,field2load,receipe_nu])
                
                # Collision time
                name = 'tcoll%s%s'%(s_,r_)
                symbol = r'$\tau^{coll}_{%s%s}$'%(s_,r_)
                units = r's'
                field2load = ['n%s'%(s_), 'Tpar%s'%(s_), 'Tperp%s'%(s_),
                              'n%s'%(r_), 'Tpar%s'%(r_), 'Tperp%s'%(r_),
                              'Bmag']
                def receipe_tcoll(gdata_list,qs=spec.q,ms=spec.m,qr=rpec.q,mr=rpec.m):
                    return 1/receipe_nu(gdata_list,qs=qs,ms=ms,qr=qr,mr=mr)
                default_qttes.append([name,symbol,units,field2load,receipe_tcoll])

            #- The following are vector fields quantities that we treat component wise
            directions = ['x','y','z'] #directions array
            for i_ in range(len(directions)):
                ci_ = directions[i_] # direction of the flux component
                cj_ = directions[np.mod(i_+1,3)] # direction coord + 1
                ck_ = directions[np.mod(i_+2,3)] # direction coord + 2
                
                # Diamagnetic drift velocity \mathbf{v}{Ds} = \frac{1}{q n B} \, \mathbf{b} \times \nabla p\perp
                name       = 'dia_v_%s%s'%(ci_,s_)
                symbol     = r'$v_{D%s %s}$'%(ci_,s_)
                units      = r'm/s'
                field2load = ['n%s'%s_,'Tperp%s'%s_,'b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian']
                def receipe_vDia(gdata_list,i=i_,q=spec.q,m=spec.m):
                    density = pgkyl_.get_values(gdata_list[0])
                    Tperp   = pgkyl_.get_values(gdata_list[1])*m
                    b_j     = pgkyl_.get_values(gdata_list[2])
                    b_k     = pgkyl_.get_values(gdata_list[3])
                    Bmag    = pgkyl_.get_values(gdata_list[4])
                    Jacob   = pgkyl_.get_values(gdata_list[5])
                    grids   = pgkyl_.get_grid(gdata_list[0])
                    j       = np.mod(i+1,3)
                    k       = np.mod(i+2,3)
                    jgrid   = grids[j][:-1]
                    kgrid   = grids[k][:-1]
                    dpdj  = np.gradient(density*Tperp, jgrid, axis=j)
                    dpdk  = np.gradient(density*Tperp, kgrid, axis=k)
                    return 1.0/(q * density * Bmag) * (b_j*dpdk - b_k*dpdj)/Jacob
                default_qttes.append([name,symbol,units,field2load,receipe_vDia])
                
                # gradB drift velocity \mathbf{v}{\nabla B} = \frac{m v\perp^2}{2 q B^2} \, \mathbf{b} \times \nabla B
                name       = 'gradB_v_%s%s'%(ci_,s_)
                symbol     = r'$u_{\nabla B,%s %s}$'%(ci_,s_)
                units      = r'm/s'
                field2load = ['Tperp%s'%s_,'b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian']
                def receipe_vgB(gdata_list,i=i_,q=spec.q,m=spec.m):
                    Tperp   = pgkyl_.get_values(gdata_list[0])*m
                    b_j     = pgkyl_.get_values(gdata_list[1])
                    b_k     = pgkyl_.get_values(gdata_list[2])
                    Bmag    = pgkyl_.get_values(gdata_list[3])
                    Jacob   = pgkyl_.get_values(gdata_list[4])
                    grids   = pgkyl_.get_grid(gdata_list[0])
                    j       = np.mod(i+1,3)
                    k       = np.mod(i+2,3)
                    jgrid   = grids[j][:-1]
                    kgrid   = grids[k][:-1]
                    dBdj  = np.gradient(Bmag, jgrid, axis=j)
                    dBdk  = np.gradient(Bmag, kgrid, axis=k)
                    return Tperp/q * (b_j*dBdk - b_k*dBdj)/(Jacob * Bmag**2)
                default_qttes.append([name,symbol,units,field2load,receipe_vgB])

                # ExB particle flux
                name       = 'ExB_pflux_%s%s'%(ci_,s_)
                symbol     = r'$\Gamma_{E%s,%s}$'%(ci_,s_)
                units      = r's$^{-1}$m$^{-2}$'
                field2load = ['n%s'%s_,'phi','b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian']
                # The receipe depends on the direction 
                # because of the phi derivative
                def receipe_ExB_pflux_s(gdata_list,i=i_):
                    density = pgkyl_.get_values(gdata_list[0])
                    vE      = receipe_vExB(gdata_list[1:],i=i)
                    return density*vE
                
                default_qttes.append([name,symbol,units,field2load,receipe_ExB_pflux_s])

                # ExB heat fluxes
                name       = 'ExB_hflux_%s%s'%(ci_,s_)
                symbol     = r'$Q_{E%s,%s}$'%(ci_,s_)
                units       = r'J s$^{-1}$m$^{-2}$'
                field2load = ['n%s'%s_,'Tpar%s'%s_,'Tperp%s'%s_,'phi',
                              'b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian',]
                # The receipe depends on the direction 
                # because of the phi derivative and on the species 
                # (temperature from J/kg to J)
                def receipe_ExB_hflux_s(gdata_list,i=i_,m=spec.m):
                    density = pgkyl_.get_values(gdata_list[0])
                    Ttot    = receipe_Ttots(gdata_list[1:3])
                    vE      = receipe_vExB(gdata_list[3:],i=i)
                    return density * m*Ttot * vE
                default_qttes.append([name,symbol,units,field2load,receipe_ExB_hflux_s])

                # gradB particle flux
                name       = 'gradB_pflux_%s%s'%(ci_,s_)
                symbol     = r'$\Gamma_{\nabla B%s,%s}$'%(ci_,s_)
                units      = r's$^{-1}$m$^{-2}$'
                field2load = ['n%s'%s_,'Tperp%s'%s_,'b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian']
                # The receipe depends on the direction 
                # because of the phi derivative
                def receipe_gradB_pflux_s(gdata_list,i=i_,q=spec.q,m=spec.m):
                    density = pgkyl_.get_values(gdata_list[0])
                    vgB     = receipe_vgB(gdata_list[1:],i=i,q=q,m=m)
                    return density*vgB
                default_qttes.append([name,symbol,units,field2load,receipe_gradB_pflux_s])
                
                # gradB heat flux
                name       = 'gradB_hflux_%s%s'%(ci_,s_)
                symbol     = r'$Q_{\nabla B%s,%s}$'%(ci_,s_)
                units      = r'J s$^{-1}$m$^{-2}$'
                field2load = ['n%s'%s_,'Tpar%s'%s_,'Tperp%s'%s_,
                              'b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian']
                # The receipe depends on the direction 
                # because of the phi derivative
                def receipe_gradB_hflux_s(gdata_list,i=i_,q=spec.q,m=spec.m):
                    density = pgkyl_.get_values(gdata_list[0])
                    Ttot    = receipe_Ttots(gdata_list[1:3])
                    vgB     = receipe_vgB(gdata_list[2:],i=i,q=q,m=m)
                    return density * m*Ttot * vgB
                default_qttes.append([name,symbol,units,field2load,receipe_gradB_hflux_s])
                
                # speciewise total particle flux
                name       = 'pflux_%s%s'%(ci_,s_)
                symbol     = r'$\Gamma_{%s,%s}$'%(ci_,s_)
                units      = r's$^{-1}$m$^{-2}$'
                field2load = ['n%s'%s_,'Tperp%s'%(s_),'phi',
                              'b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian']
                def receipe_pflux_s(gdata_list,i=i_,q=spec.q,m=spec.m):
                    pExBlist = gdata_list.copy()
                    pExBlist.pop(1) # remove Tperp
                    pExB = receipe_ExB_pflux_s(pExBlist,i=i)
                    pgBlist = gdata_list.copy()
                    pgBlist.pop(2) # remove phi
                    pgB   = receipe_gradB_pflux_s(pgBlist,i=i,q=q,m=m)
                    return pExB + pgB
                default_qttes.append([name,symbol,units,field2load,receipe_pflux_s])
                
                # speciewise total heat flux
                name       = 'hflux_%s%s'%(ci_,s_)
                symbol     = r'$Q_{%s,%s}$'%(ci_,s_)
                units      = r'J s$^{-1}$m$^{-2}$'
                field2load = ['n%s'%s_,'Tpar%s'%(s_),'Tperp%s'%(s_),'phi',
                              'b_%s'%cj_,'b_%s'%ck_,'Bmag','Jacobian']
                def receipe_hflux_s(gdata_list,i=i_,q=spec.q,m=spec.m):
                    QExB = receipe_ExB_hflux_s(gdata_list,i=i)
                    QgBlist = gdata_list.copy()
                    QgBlist.pop(3) # remove phi
                    QgB   = receipe_gradB_hflux_s(QgBlist,i=i,q=q,m=m)
                    return QExB + QgB
                default_qttes.append([name,symbol,units,field2load,receipe_hflux_s])

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

        #Debye length (lambda_D = sqrt(e0 kB Te / sum_s(q_s^2 n_s)))
        name = 'lambdaD'
        symbol = r'$\lambda_{D}$'
        units = r'm'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('n%s'%s_)
            field2load.append('Tpar%s'%s_)
            field2load.append('Tperp%s'%s_)
        def receipe_lambdaD(gdata_list,species=species):
            e = 1.602176634e-19
            denom = 0.0
            i_s = 0
            for spec in species.values():
                n_s = pgkyl_.get_values(gdata_list[0+i_s])
                T_s = receipe_Ttots([gdata_list[1+i_s],gdata_list[2+i_s]])
                T_s[T_s <= 0] = np.nan # avoid unphysical values
                denom += (spec.q/e)**2 * n_s/T_s
                i_s += 3
            e0 = 8.854187817e-12
            num = e0 * phys_tools.kB / e**2
            denom[denom <= 0] = np.nan # avoid div by 0
            return np.sqrt(num/denom)
        default_qttes.append([name,symbol,units,field2load,receipe_lambdaD])
        
        #electron larmor radius to Debye length ratio
        name = 'rhoe_lambdaD'
        symbol = r'$\rho_{e}/\lambda_{D}$'
        units = ''
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('n%s'%s_)
            field2load.append('Tpar%s'%s_)
            field2load.append('Tperp%s'%s_)
        field2load.append('Tperpe')
        field2load.append('Bmag')
        def receipe_rhoe_lambdaD(gdata_list,species=species):
            lambdaD = receipe_lambdaD(gdata_list,species=species)
            rho_e = receipe_rhos(gdata_list[-2:],q=species['elc'].q,m=species['elc'].m)
            return rho_e/lambdaD
        default_qttes.append([name,symbol,units,field2load,receipe_rhoe_lambdaD])
        
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
                fout += spec.q*pgkyl_.get_values(gdata_list[0+k])*pgkyl_.get_values(gdata_list[1+k])
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
            
            # Electrostatic Electric field i-th component
            name       = 'E%s_es'%(ci_)
            symbol     = r'$E_{%s}^{es}$'%(ci_)
            units      = r'V/m'
            field2load = ['phi']
            def receipe_Ei_es(gdata_list,i=i_):
                phi     = pgkyl_.get_values(gdata_list[0])
                grids   = pgkyl_.get_grid(gdata_list[0])
                igrid    = grids[i][:-1]
                return -np.gradient(phi, igrid, axis=i)
            default_qttes.append([name,symbol,units,field2load,receipe_Ei_es])
            
            # EM Electric field i-th component
            name       = 'E%s'%(ci_)
            symbol     = r'$E_{%s}$'%(ci_)
            units      = r'V/m'
            field2load = ['phi','Apardot']
            def receipe_Ei(gdata_list,i=i_,ci=ci_):
                if ci == 'z':
                    Apardot = pgkyl_.get_values(gdata_list[1])
                    return receipe_Ei_es(gdata_list[0:1],i=i) - Apardot
                else:
                    return receipe_Ei_es(gdata_list[0:1],i=i)
            default_qttes.append([name,symbol,units,field2load,receipe_Ei])
            
        #total source power density
        name       = 'src_P'
        symbol     = r'$P_{src}$'
        units      = r'W/m$^3$'
        field2load = []
        for spec in species.values():
            s_ = spec.nshort
            field2load.append('src_HM_H%s'%(s_))
            field2load.append('phi')
            field2load.append('src_n%s'%(s_))
                            
        def receipe_src_pow(gdata_list,species=species):
            fout = 0.0
            k    = 0
            for spec in species.values():
                fout += receipe_src_Ps(gdata_list[0+k:3+k],q=spec.q)
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

        directions = ['x','y','z'] #directions array
        for i_ in range(len(directions)):
            ci_ = directions[i_] # direction of the flux component
            cj_ = directions[np.mod(i_+1,3)] # direction coord + 1
            ck_ = directions[np.mod(i+2,3)] # direction coord + 2  
                      
            # total ExB particle flux: \Gamma_ExB = \sum_s \Gamma_ExB_s
            name      = 'ExB_pflux_%s'%(ci_)
            symbol    = r'$\Gamma_{%s}$'%(ci_)
            units     = r's$^{-1}$m$^{-2}$'
            field2load = []
            for spec in species.values():
                s_ = spec.nshort
                field2load.append('n%s'%s_)
                field2load.append('phi')
                field2load.append('b_%s'%cj_)
                field2load.append('b_%s'%ck_)
                field2load.append('Bmag')
                field2load.append('Jacobian')
            def receipe_ExB_pflux(gdata_list,i=i_,species=species):
                fout = 0.0
                # add species dependent energies
                k = 0
                for spec in species.values():
                    fout += receipe_ExB_pflux_s(gdata_list[0+k:6+k], i=i)
                    k+= 6
                return fout
            default_qttes.append([name,symbol,units,field2load,receipe_ExB_pflux])
            
            #total ExB heat flux: Q_ExB = \sum_s Q_ExB_s
            name       = 'ExB_hflux_%s'%(ci_)
            symbol     = r'$Q_{%s}$'%(ci_)
            units      = r'J s$^{-1}$m$^{-2}$'
            field2load = []
            for spec in species.values():
                s_ = spec.nshort
                field2load.append('n%s'%s_)
                field2load.append('Tpar%s'%s_)
                field2load.append('Tperp%s'%s_)
                field2load.append('phi')
                field2load.append('b_%s'%cj_)
                field2load.append('b_%s'%ck_)
                field2load.append('Bmag')
                field2load.append('Jacobian')
            def receipe_ExB_hflux(gdata_list,i=i_,species=species):
                fout = 0.0
                # add species dependent energies
                k = 0
                for spec in species.values():
                    fout += receipe_ExB_hflux_s(gdata_list[0+k:8+k], i=i, m=spec.m)
                    k+= 8
                return fout
            default_qttes.append([name,symbol,units,field2load,receipe_ExB_hflux]) 
            
            #total gradB particle flux: \Gamma_gradB = \sum_s \Gamma_gradB_s
            name       = 'gradB_pflux_%s'%(ci_)
            symbol     = r'$\Gamma_{\nabla B%s}$'%(ci_)
            units      = r's$^{-1}$m$^{-2}$'
            field2load = []
            for spec in species.values():
                s_ = spec.nshort
                field2load.append('n%s'%s_)
                field2load.append('Tperp%s'%s_)
                field2load.append('b_%s'%cj_)
                field2load.append('b_%s'%ck_)
                field2load.append('Bmag')
                field2load.append('Jacobian')
            def receipe_gradB_pflux(gdata_list,i=i_,species=species):
                fout = 0.0
                # add species dependent energies
                k = 0
                for spec in species.values():
                    fout += receipe_gradB_pflux_s(gdata_list[0+k:6+k], i=i, q=spec.q, m=spec.m)
                    k+= 6
                return fout
            default_qttes.append([name,symbol,units,field2load,receipe_gradB_pflux])
            
            #total gradB heat flux: Q_gradB = \sum_s Q_gradB_s
            name       = 'gradB_hflux_%s'%(ci_)
            symbol     = r'$Q_{\nabla B%s}$'%(ci_)
            units      = r'J s$^{-1}$m$^{-2}$'
            field2load = []
            for spec in species.values():
                s_ = spec.nshort
                field2load.append('n%s'%s_)
                field2load.append('Tpar%s'%s_)
                field2load.append('Tperp%s'%s_)
                field2load.append('b_%s'%cj_)
                field2load.append('b_%s'%ck_)
                field2load.append('Bmag')
                field2load.append('Jacobian')
            def receipe_gradB_hflux(gdata_list,i=i_,species=species):
                fout = 0.0
                # add species dependent energies
                k = 0
                for spec in species.values():
                    fout += receipe_gradB_hflux_s(gdata_list[0+k:8+k], i=i, q=spec.q, m=spec.m)
                    k+= 8
                return fout
            default_qttes.append([name,symbol,units,field2load,receipe_gradB_hflux])
            
            #total particle flux: \Gamma = \sum_s \Gamma_s
            name       = 'pflux_%s'%(ci_)
            symbol     = r'$\Gamma_{%s}$'%(ci_)
            units      = r's$^{-1}$m$^{-2}$'
            field2load = []
            for spec in species.values():
                s_ = spec.nshort
                field2load.append('n%s'%s_)
                field2load.append('Tperp%s'%(s_))
                field2load.append('phi')
                field2load.append('b_%s'%cj_)
                field2load.append('b_%s'%ck_)
                field2load.append('Bmag')
                field2load.append('Jacobian')
            def receipe_pflux(gdata_list,i=i_,species=species):
                fout = 0.0
                # add species dependent energies
                k = 0
                for spec in species.values():
                    fout += receipe_pflux_s(gdata_list[0+k:7+k], i=i, q=spec.q, m=spec.m)
                    k+= 6
                return fout
            default_qttes.append([name,symbol,units,field2load,receipe_pflux])
            
            #total heat flux: Q = \sum_s Q_s
            name       = 'hflux_%s'%(ci_)
            symbol     = r'$Q_{%s}$'%(ci_)
            units      = r'J s$^{-1}$m$^{-2}$'
            field2load = []
            for spec in species.values():
                s_ = spec.nshort
                field2load.append('n%s'%s_)
                field2load.append('Tpar%s'%(s_))
                field2load.append('Tperp%s'%(s_))
                field2load.append('phi')
                field2load.append('b_%s'%cj_)
                field2load.append('b_%s'%ck_)
                field2load.append('Bmag')
                field2load.append('Jacobian')
            def receipe_hflux(gdata_list,i=i_,species=species):
                fout = 0.0
                # add species dependent energies
                k = 0
                for spec in species.values():
                    fout += receipe_hflux_s(gdata_list[0+k:8+k], i=i, q=spec.q, m=spec.m)
                    k+= 8
                return fout
            default_qttes.append([name,symbol,units,field2load,receipe_hflux])
                    
        #--- Flan interface
        def receipe_flan(gdata_list): return
        name = 'flan_imp_density'
        symbol = r'$n_{Z}$'
        units = r'm$^{-3}$'
        field2load = ['flan'] # phi is here just to get conf grids info, the flan interface will get the values
        default_qttes.append([name,symbol,units,field2load,receipe_flan])
        
        name = 'flan_imp_counts'
        symbol = r'$N_{Z}$'
        units = r''
        field2load = ['flan'] # phi is here just to get conf grids info, the flan interface will get the values
        default_qttes.append([name,symbol,units,field2load,receipe_flan])

        name = 'flan_imp_gyrorad'
        symbol = r'$\rho_{W}$'
        units = r'm'
        field2load = ['flan'] # phi is here just to get conf grids info, the flan interface will get the values
        default_qttes.append([name,symbol,units,field2load,receipe_flan])
        
        name = 'flan_electron_dens'
        symbol = r'$n_{e}$'
        units = r'm$^{-3}$'
        field2load = ['flan'] # phi is here just to get conf grids info, the flan interface will get the values
        default_qttes.append([name,symbol,units,field2load,receipe_flan])

        name = 'flan_electron_temp'
        symbol = r'$T_{e}$'
        units = r'eV'
        field2load = ['flan'] # phi is here just to get conf grids info, the flan interface will get the values
        default_qttes.append([name,symbol,units,field2load,receipe_flan])
        
        name = 'flan_ion_temp'
        symbol = r'$T_{i}$'
        units = r'eV'
        field2load = ['flan'] # phi is here just to get conf grids info, the flan interface will get the values
        default_qttes.append([name,symbol,units,field2load,receipe_flan])

        name = 'flan_plasma_pot'
        symbol = r'$V_{p}$'
        units = r'V'
        field2load = ['flan'] # phi is here just to get conf grids info, the flan interface will get the values
        default_qttes.append([name,symbol,units,field2load,receipe_flan])

        name = 'flan_bmag_R'
        symbol = r'$B_{R}$'
        units = r'T'
        field2load = ['flan'] # phi is here just to get conf grids info, the flan interface will get the values
        default_qttes.append([name,symbol,units,field2load,receipe_flan])

        dirs = ['x','y','z']
        Dirs = ['X','Y','Z']
        for i in range(3):
            dir = dirs[i]
            Dir = Dirs[i]
            field2load = ['flan'] # phi is here just to get conf grids info, the flan interface will get the values

            # Cartesian velocity components
            name = 'flan_imp_v'+Dir
            symbol = r'$v_{Z,%s}$'%dir
            units = r'm/s'
            default_qttes.append([name,symbol,units,field2load,receipe_flan])

            # Cartesian electric field components
            name = 'flan_elec_'+Dir
            symbol = r'$E_{%s}$'%dir
            units = r'V/m'
            default_qttes.append([name,symbol,units,field2load,receipe_flan])

            # Cartesian ion velocity components
            name = 'flan_ion_flow_'+Dir
            symbol = r'$u_{%s}$'%dir
            units = r'm/s'
            default_qttes.append([name,symbol,units,field2load,receipe_flan])

            # Cartesian magentic field components
            name = 'flan_bmag_'+Dir
            symbol = r'$B_{%s}$'%dir
            units = r'T'
            default_qttes.append([name,symbol,units,field2load,receipe_flan])

            # Cartesian magentic field components
            name = 'flan_gradb_'+Dir
            symbol = r'$\nabla B_{%s}$'%dir
            units = r'T/m'
            default_qttes.append([name,symbol,units,field2load,receipe_flan])

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

        # add default colormap for each fields
        positive_fields = ['Bmag','pow_src','rhoe_lambdaD',
                           'flan_imp_density','flan_imp_counts'] # spec. indep
        
        spec_dep_fields = ['M0','M2','M2par','M2perp',
                           'n','T','Tpar','Tperp','p',
                           'BM_n','BM_Tpar','BM_Tperp',
                           'MM_n','MM_T', 'HM_n','HM_H',
                           'src_M0','src_M2','src_M2par','src_M2perp',
                           'src_n','src_T','src_Tpar','src_Tperp','src_p',
                           'src_BM_n','src_BM_Tpar','src_BM_Tperp',
                            'src_MM_n','src_MM_T','src_HM_n','src_HM_H',
                           'f','src_f','rho','lambdaD']
        for sdepfield in spec_dep_fields:
            for spec in species.values():
                positive_fields.append(sdepfield+spec.nshort)
                
        double_spec_fields = ['nu','tcoll']
        for sdepfield in double_spec_fields:
            for spec in species.values():
                for rpec in species.values():
                    positive_fields.append(sdepfield+spec.nshort+rpec.nshort)
        
        for field in names:
            if field in positive_fields:
                default_units_dict[field+'colormap'] = 'inferno'
            else:
                default_units_dict[field+'colormap'] = 'bwr'
                
        # Sort the dictionary by keys
        default_units_dict = dict(sorted(default_units_dict.items()))
        
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
