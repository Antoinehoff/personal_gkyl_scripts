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

        # The radial (cross-field, not cylindrical R) and poloidal velocities
        elif dname.lower() in ["v_rad", "v_pol", "exb_rad", "exb_pol"]:

            # Geometry arrays (nx, ny, nz)
            X = flan_nc["geometry"]["X"][:].data
            Y = flan_nc["geometry"]["Y"][:].data
            Z = flan_nc["geometry"]["Z"][:].data
            R = np.sqrt(X**2 + Y**2)

            # Magnetic field arrays
            BX = flan_nc["background"]["B_X"][tframe]   # (nx, ny, nz)
            BY = flan_nc["background"]["B_Y"][tframe]
            BZ = flan_nc["background"]["B_Z"][tframe]

            # Velocity components
            if dname.lower() in ["v_rad", "v_pol"]:
                vX = flan_nc["output"]["v_X"][tframe]   # (nx, ny, nz)
                vY = flan_nc["output"]["v_Y"][tframe]
                vZ = flan_nc["output"]["v_Z"][tframe]

            # In theory straightforward to implement...
            if dname.lower() in ["exb_rad", "exb_pol"]:
                print("ExB radial/poloidal components not implemented.")

            # Stack into vector fields
            B = np.stack([BX, BY, BZ], axis=-1)         # (nx, ny, nz, 3)
            v = np.stack([vX, vY, vZ], axis=-1)

            # Toroidal unit vector e_phi = (-y, x, 0) / R
            e_phi = np.zeros(X.shape + (3,))
            e_phi[..., 0] = -Y / R
            e_phi[..., 1] =  X / R
            e_phi[..., 2] =  0.0

            # Normalize
            e_phi /= np.linalg.norm(e_phi, axis=-1, keepdims=True)

            # Broadcast e_phi to match time dimension. Broadcast requires 
            # the rightmost index match, which in this case is 3 for both 
            # arrays. Now e_phi behaves like an (nt, nx, ny, nz, 3) shaped 
#               # array without actually using any additional memory.
            e_phi = np.broadcast_to(e_phi, B.shape)

            # Poloidal direction = B x e_phi
            e_pol = np.cross(B, e_phi)
            e_pol /= np.linalg.norm(e_pol, axis=-1, keepdims=True)
            
            # Radial direction = e_phi x e_pol
            e_r = np.cross(e_phi, e_pol)
            e_r /= np.linalg.norm(e_r, axis=-1, keepdims=True)

            # Project velocity onto radial and poloidal directions. This is just
            # the dot product. v*e_r is the 3 terms in the dot product, then we
            # sum them along the coordinate axes to finish the dot product.
            if dname.lower() == "v_rad":
                return np.sum(v * e_r, axis=-1)
            elif dname.lower() == "v_pol":
                return np.sum(v * e_pol, axis=-1)


    def load_data(self, fieldname, tframe, xyz=True):

        # Create actual data name string and load netCDF file
        dname = fieldname.replace('flan_', '')

        #flan_nc = netCDF4.Dataset(self.path, 'r')
        with netCDF4.Dataset(self.path, "r") as flan_nc:

            # If the data is considered a derived value, then call 
            # load_derived_values to perform the calculations, otherwise just load
            # the data straight up.
            derived_dnames = ["gca_validity_time_EX", "gca_validity_time_EY", 
                "gca_validity_time_EZ", "gca_validity_time_E", "v_rad", "v_pol"]
            if dname in derived_dnames:
                values = self.load_derived_values(flan_nc, dname, tframe)
            else:

              # Return variable name from whichever group it's in
              for group in ["output", "background", "geometry"]:
                if dname in flan_nc[group].variables.keys():
                  values = flan_nc[group][dname][tframe].data
            
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
    
