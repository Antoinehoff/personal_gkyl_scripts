# DataParam.py
class DataParam:
    def __init__(self, expdatadir='', g0simdir='', simname='', simdir='', fileprefix='', wkdir=''):
        self.expdatadir = expdatadir
        self.g0simdir = g0simdir
        self.simname = simname
        self.simdir = g0simdir+simdir
        self.wkdir = wkdir
        self.datadir = g0simdir + simdir + simname +'/' + wkdir
        self.fileprefix = self.datadir + fileprefix
        self.data_files_dict = {}
        self.set_data_field_dict()

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
        
    def set_data_field_dict(self,keys=[],files=[],comps=[]):
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
        # add elc moments info
        keys = ['ne','upare','Tpare','Tperpe']
        comps= [   0,      1,      2,       3]
        for i in range(len(keys)):
            data_field_dict[keys[i]+'file'] = 'elc_BiMaxwellianMoments' 
            data_field_dict[keys[i]+'comp'] = comps[i]
            data_field_dict[keys[i]+'gnames'] = gnames[0:3]
        # add ion moments info
        keys = ['ni','upari','Tpari','Tperpi']
        comps= [   0,      1,      2,       3]
        for i in range(len(keys)):
            data_field_dict[keys[i]+'file'] = 'ion_BiMaxwellianMoments' 
            data_field_dict[keys[i]+'comp'] = comps[i]
            data_field_dict[keys[i]+'gnames'] = gnames[0:3]
        # add distribution functions
        data_field_dict['fe'+'file'] = 'elc'
        data_field_dict['fe'+'comp'] = 0
        data_field_dict['fe'+'gnames'] = gnames
        
        data_field_dict['fi'+'file'] = 'ion'
        data_field_dict['fi'+'comp'] = 0
        data_field_dict['fi'+'gnames'] = gnames

        self.data_files_dict = data_field_dict