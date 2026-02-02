import netCDF4
import os
import numpy as np


# Define some constants as global variables
amu_to_kg = 1.66e-27
elec = -1.602e-19

class FlanInterface:
    
    def __init__(self,path):
        self.path = path
        if self.path is None:
            self.avail_frames = []
        elif not os.path.exists(self.path):
            raise FileNotFoundError(f"Flan data file {self.path} not found.") 
        else:      
            flan_nc = netCDF4.Dataset(self.path, 'r')
            self.avail_frames = [i for i in range(len(flan_nc["geometry"]['time'][:].data))]
        

    def calc_imp_cycl_freq(self, flan_nc, tframe):
        """
        Helper function to calculate impurity cyclotron frequency
        """

        # Pull out required arrays
        BX = flan_nc["background"]["B_X"][tframe].data
        BY = flan_nc["background"]["B_Y"][tframe].data
        BZ = flan_nc["background"]["B_Z"][tframe].data
        mz_kg = flan_nc["input"]["mz"][:].data * amu_to_kg
        qz = flan_nc["output"]["qz"][tframe].data  # Average charge state

        # Magnetic field magnitude squared
        Bsq = np.square(BX) + np.square(BY) + np.square(BZ)

        # Cyclotron frequency
        omega_c = np.abs(qz * elec) * np.sqrt(Bsq) / mz_kg
        return omega_c

    def load_derived_values(self, flan_nc, dname, tframe):
        """
        Load values from Flan that are a derived/calculated from various
        outputs. Example could be the ExB or grad-B drifts, or special
        metrics such as determining if the guiding center approximation is 
        valid.
        """

        # Guiding-center approximation validity in time using EX. Calculate
        # characteristic time of electric field changes, 
        # tau_EX = |EX / (dEX/dt)|, divided by the Larmor period of the
        # impurity followed by Flan, tau_imp = 1 / cycl_freq. When 
        # tau_EX / tau_imp <~ 10, the GCA is violated.
        if dname in ["gca_validity_time_EX", "gca_validity_time_EY", 
            "gca_validity_time_EZ", "gca_validity_time_E"]:

            # Pull out some required arrays
            time = flan_nc["time"][:].data
            if (dname == "gca_validity_time_EX"):
                Ecomp = flan_nc["background"]["E_X"][tframe].data
            elif (dname == "gca_validity_time_EY"):
                Ecomp = flan_nc["background"]["E_Y"][tframe].data
            elif (dname == "gca_validity_time_EZ"):
                Ecomp = flan_nc["background"]["E_Z"][tframe].data
            elif (dname == "gca_validity_time_E"):
                EX = flan_nc["background"]["E_X"][tframe].data
                EY = flan_nc["background"]["E_Y"][tframe].data
                EZ = flan_nc["background"]["E_Z"][tframe].data
                Ecomp = np.sqrt(np.square(EX) + np.square(EY) + np.square(EZ))

            # Create return array
            time_ratio = np.zeros(Ecomp.shape)

            # If first frame, return zeros since we rely on previous frame for
            # time derivative
            if (tframe == 0): return time_ratio

            # Time derivative
            dt = time[tframe] - time[tframe-1]
            dEcomp_dt = (Ecomp[tframe] - Ecomp[tframe-1]) / dt

            # Cyclotron frequency
            omega_c = self.calc_imp_cycl_freq(flan_nc, tframe)

            # Return tau_Ecomp / tau_imp
            return np.abs(Ecomp / dEcomp_dt) * omega_c
            

    def load_data(self, fieldname, tframe, xyz=True):

        # Create actual data name string and load netCDF file
        dname = fieldname.replace('flan_', '')
        flan_nc = netCDF4.Dataset(self.path, 'r')

        # If the data is considered a derived value, then call 
        # load_derived_values to perform the calculations, otherwise just load
        # the data straight up.
        derived_dnames = ["gca_validity_time_EX", "gca_validity_time_EY", 
            "gca_validity_time_EZ", "gca_validity_time_E"]
        if dname in derived_dnames:
            values = self.load_derived_values(flan_nc, dname, tframe)
        else:

          # Return variable name from whichever group it's in
          for group in ["output", "background", "geometry"]:
            if dname in flan_nc[group].variables.keys():
              values = flan_nc[group][dname][tframe].data
              print("{}: {} (group) {} (tframe) loaded".format(dname, group, tframe))
        
        time = float(flan_nc["geometry"]['time'][tframe].data)

        # get x grid and convert to default dtype instead of float32
        xc = flan_nc["geometry"]['x'][:].data
        yc = flan_nc["geometry"]['y'][:].data
        zc = flan_nc["geometry"]['z'][:].data

        # evaluate grid at nodal points (i.e. add a point at the end of the grid)
        x = np.append(xc, xc[-1] + (xc[-1] - xc[-2]))
        y = yc
        z = zc
        
        grids = [x.astype(float), y.astype(float), z.astype(float)]
        
        jacobian = flan_nc["geometry"]['J'][:].data.astype(float)
        
        jacobian = 0.5* (jacobian[1:,1:,1:] + jacobian[:-1,:-1,:-1])
        
        return time, grids, jacobian, values
    
