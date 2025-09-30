import postgkyl as pg
import numpy as np
from ..interfaces import pgkyl_interface as pgkyl_
from ..tools import math_tools as mt
import os

class GeomParam:
    """
    Manages geometric parameters like axis positions, LCFS, and safety factor profiles.

    Methods:
    - __init__: Initializes the GeomParam object with the required parameters.
    - load_metric: Loads the metric data from files.
    - r_x: Calculates the minor radius as a function of x.
    - compute_bxgradBoB2: Computes the cross product of bmag and gradB divided by Jacobian and Bmag^2.
    - set_domain: Sets the simulation domain based on the specified geometry type.
    - qprofile_default: Default safety factor profile.
    - R_f: Calculates the major radius as a function of r and theta.
    - Z_f: Calculates the height as a function of r and theta.
    - R_f_r: Calculates the derivative of R_f with respect to r.
    - R_f_theta: Calculates the derivative of R_f with respect to theta.
    - Z_f_r: Calculates the derivative of Z_f with respect to r.
    - Z_f_theta: Calculates the derivative of Z_f with respect to theta.
    - Jr_f: Calculates the Jacobian determinant.
    - integrand: Integrand for the flux surface integral.
    - dPsidr_f: Calculates the derivative of the poloidal flux with respect to r.
    - alpha_f: Calculates the alpha parameter for the flux surface.
    - wrap: Wraps a value to a specified range.
    - get_conf_grid: Returns the configuration grid.
    - get_toroidal_mode_number: Calculates and returns the toroidal mode number.
    - get_epsilon: Calculates and returns the inverse aspect ratio.
    - info: Prints the geometric parameters.

    """
    def __init__(self, R_axis=0.0, Z_axis=0.0, R_LCFSmid=0.0, B_axis=1.4, x_out = 0.08,
                 a_shift=0.0, kappa=1.0, delta=0.0, x_LCFS=0.0, geom_type='Miller', qprofile_R='default', 
                 qfit = [], cdim = 3):
        self.B_axis     = B_axis
        self.R_axis     = R_axis
        self.Z_axis     = Z_axis
        self.a_shift    = a_shift
        self.kappa      = kappa
        self.delta      = delta
        self.x_LCFS     = x_LCFS
        self.x_out      = x_out
        self.x_in       = x_LCFS
        self.R_LCFSmid  = R_LCFSmid
        self.geom_type  = geom_type
        self.qprofile_R = qprofile_R
        self.qfit       = qfit
        self.cdim       = cdim # configuration space dimension
        self.vol_frac   = None # Volume fraction
        self.tor_num    = None # Toroidal mode number
        
        self.update_geom_params()

        self.g_ij       = None
        self.gij        = None
        self.bmag       = None
        self.b_i        = None
        self.grids      = None
        self.Jacobian   = None
        self.intJac     = None
        self.dBdx       = None
        self.dBdy       = None
        self.dBdz       = None
        self.bxgradBoB2 = None
        self.x          = None
        self.Lx         = None
        self.y          = None
        self.Ly         = None
        self.z          = None # z-grid
        self.Lz         = None # z box size
        self.n0         = None # Toroidal mode number
        self.vessel_data = None # Data for the vessel corners
        self.camera_global = None # Camera settings for visualization
        self.camera_zoom_lower = None # Camera settings for lower zoom level
        self.camera_zoom_obmp = None # Camera settings for obmp zoom level
        self.camera_global_2by1 = None # Camera settings for 2by1 zoom level
        self.camera_zoom_2by1 = None # Camera settings for lower zoom level in 2by1 mode
        self.camera_poloidal = None # Camera settings for poloidal view
        
    def change(self,**kwargs):
        """
        Change the parameters of the GeomParam object.
        example:
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"GeomParam has no attribute '{key}'")
        self.update_geom_params()
        
    def update_geom_params(self):
        self.Rmid_min   = self.R_LCFSmid - self.x_in # Minimum midplane major radius of simulation box [m].
        self.Rmid_max   = self.R_LCFSmid + self.x_out  # Maximum midplane major radius of simulation box [m].
        self.R0         = 0.5*(self.Rmid_min + self.Rmid_max)  # Major radius of the simulation box [m].
        self.r0         = self.R0 - self.R_axis          # Minor radius of the simulation box [m].
        self.B0         = self.B_axis * (self.R_axis/self.R0) # Magnetic field magnitude in the simulation box [T].
        if self.a_shift > 0.0:
            self.a_mid      = self.R_axis/self.a_shift - \
                np.sqrt(self.R_axis*(self.R_axis - 2*self.a_shift*self.R_LCFSmid + 2*self.a_shift*self.R_axis))\
                    /self.a_shift
        else:
            self.a_mid      = self.R_LCFSmid - self.R_axis

        if callable(self.qprofile_R):
            self.qprofile_R = self.qprofile_R
        else:
            if self.qfit is None:
                self.qfit = [497.3420166252413,-1408.736172826569,1331.4134861681464,-419.00692601227627]
            self.qprofile_R = lambda R: np.polyval(self.qfit, R)
        self.q0 = self.qprofile_R(self.R0)
        self.Cy = self.r0/self.q0 if self.q0 != 0 else 0.0
        
    def load_metric(self,fileprefix):
        #-- load B (bmag)
        fname = fileprefix+'-'+'bmag.gkyl'
        # check if fname exist
        if not os.path.exists(fname):
            raise Exception(f"File {fname} does not exist, please review simDir and filePrefix.")
        Gdata = pg.data.GData(fname)
        dg = pg.data.GInterpModal(Gdata,1,'ms')
        dg.interpolate(0,overwrite=True)
        self.bmag = pgkyl_.get_values(Gdata)
        self.bmag = self.bmag[:,:,:,0]
        
        if self.bmag.shape[1] <= 2 and self.cdim == 3:
            print("Warning: it seems that the sim data are 2x2v but the simulation class was initialized with '3x2v'.")
            print("use simulation = pygkyl.simulation_configs.import_config( ...,  ..., ..., dimensionality='2x2v')")
        
        #-- load grid
        self.grids = [0.5*(g[1:]+g[:-1]) for g in pgkyl_.get_grid(Gdata) if len(g) > 1]
        self.x = self.grids[0]; self.y = self.grids[1]; self.z = self.grids[2]
        self.Lx    = self.x[-1]-self.x[0]
        self.Ly    = self.y[-1]-self.y[0]
        self.n0    = 2.*np.pi*self.r0/self.q0/self.Ly
        self.Lz    = self.z[-1]-self.z[0]

        #-- compute associated derivatives
        self.dBdx = mt.gradient(self.bmag, self.grids[0], axis=0)  # Derivative w.r.t x
        self.dBdy = mt.gradient(self.bmag, self.grids[1], axis=1)  # Derivative w.r.t y
        self.dBdz = mt.gradient(self.bmag, self.grids[2], axis=2)  # Derivative w.r.t z

        #-- Load b_x, b_y, and b_z
        fname = fileprefix+'-'+'b_i.gkyl'
        self.b_i = []
        for i in range(3):
            Gdata = pg.data.GData(fname)
            dg = pg.data.GInterpModal(Gdata,1,'ms')
            dg.interpolate(i,overwrite=True)
            tmp_ = pgkyl_.get_values(Gdata)
            self.b_i.append(tmp_[:,:,:,0])

        #-- Load Jacobian
        fname = fileprefix+'-'+'jacobgeo.gkyl'
        Gdata = pg.data.GData(fname)
        dg = pg.data.GInterpModal(Gdata,1,'ms')
        dg.interpolate(0,overwrite=True)
        self.Jacobian = pgkyl_.get_values(Gdata)
        self.Jacobian = self.Jacobian[:,:,:,0]
        J_yz          = np.trapz(self.Jacobian,self.x,axis=0)
        J_z           = np.trapz(J_yz,self.y,axis=0)
        self.intJac   = np.trapz(J_z,self.z,axis=0)
        
        #-- add a volume fraction and toroidal mode number
        if self.cdim == 2:
            self.vol_frac = 1.0/(2.*np.pi*self.R0)
        elif self.cdim == 3:
            self.vol_frac = 1.0/(2.*np.pi*self.r0/self.q0/self.Ly)

    def compute_bxgradBoB2(self):
        # The gradient of B (i.e., grad B) is a vector field
        gradB = np.array([self.dBdx, self.dBdy, self.dBdz])
        b_x = self.b_i[0]
        b_y = self.b_i[1]
        b_z = self.b_i[2]
        # Now compute the cross product bmag x gradB for each component
        # Cross product formula:
        # (Bx, By, Bz) x (dBdx, dBdy, dBdz) = (By*dBdz - Bz*dBdy, Bz*dBdx - Bx*dBdz, Bx*dBdy - By*dBdx)
        self.bxgradBoB2    = np.empty_like(gradB)
        self.bxgradBoB2[0] = b_y * self.dBdz - b_z * self.dBdy  # x-component
        self.bxgradBoB2[1] = b_z * self.dBdx - b_x * self.dBdz  # y-component
        self.bxgradBoB2[2] = b_x * self.dBdy - b_y * self.dBdx  # z-component
        # Now divide the cross product by jacobian and Bmag^2
        self.bxgradBoB2 /= self.Jacobian * self.bmag**2

    def set_domain(self,geom_type='Miller',vessel_corners=[[0.6,1.2],[-0.7,0.7]],Ntheta=128):
        if geom_type == 'Miller':
            ## Miller geometry model
            def RZ_rtheta(R_, theta, delta, kappa):
                r = R_ - self.R_axis
                R = (self.R_axis - self.a_shift * r**2 / (2.0 * self.R_axis) + \
                        r * np.cos(theta + np.arcsin(delta) * np.sin(theta)))
                Z = self.Z_axis + kappa*r*np.sin(theta)
                return [R,Z]
            Rmid_min = self.R_LCFSmid - self.x_LCFS
            Rmid_max = Rmid_min+self.Lx
            theta = np.linspace(-np.pi,+np.pi,Ntheta)
            self.RZ_min  = RZ_rtheta(Rmid_min,theta,self.delta,self.kappa)
            self.RZ_max  = RZ_rtheta(Rmid_max,theta,self.delta,self.kappa)
            self.RZ_lcfs = RZ_rtheta(self.R_LCFSmid,theta,self.delta,self.kappa)
            self.vessel_corners = vessel_corners           
        elif geom_type == 'efit':
            a = 0
            # one day... (09/20/2024)


    #.Function that wraps x to [xMin,xMax].
    def wrap(x, xMin, xMax):
        return (((x-xMin) % (xMax-xMin)) + (xMax-xMin)) % (xMax-xMin) + xMin

    #.Minor radius as a function of x:
    def r_x(self,x):
        return self.a_mid - self.x_LCFS + x

    ## Geometric relations for Miller geometry
    def R_rt(self, r, theta):
        return self.R_axis \
            - self.a_shift * r**2/(2.0*self.R_axis) \
            + r*np.cos(theta + np.arcsin(self.delta)*np.sin(theta))
            
    def R_r(self, r):
        return self.R_rt(r, 0.0)
    
    def R_x(self, x):
        return self.R_rt(self.r_x(x), 0.0)

    def Z_rt(self, r, theta):
        return self.Z_axis + self.kappa*r*np.sin(theta)
    
    #.Analytic derivatives.
    def dRdr(self, r, theta): 
        return -self.a_shift*r/self.R_axis + np.cos(theta + np.arcsin(self.delta)*np.sin(theta))
    
    def dRdt(self, r,theta): 
        return -r*(np.arcsin(self.delta)*np.cos(theta)+1.)*np.sin(np.arcsin(self.delta)*np.sin(theta)+theta)
    
    def dZdr(self, r,theta):
        return self.kappa*np.sin(theta)
    
    def dZdt(self, r,theta):
        return self.kappa*r*np.cos(theta)
    
    def J_rt(self, r, theta):
        return self.R_rt(r,theta)*(self.dRdr(r,theta)*self.dZdt(r,theta)-self.dZdr(r,theta)*self.dRdt(r,theta))

    def integrand(self, t, r):
        return self.J_rt(r,t)/np.power(self.R_rt(r,t),2)

    def dPsidr(self, r, method='trapz32'):
        integral, _ = mt.integrate(self.integrand, a=0., b=2.*np.pi, args=(r), method=method)
        R = self.R_rt(r, 0.)
        return self.B0*self.R_axis/(2.*np.pi*self.qprofile_R(R))*integral

    def alpha(self, r, theta, phi, method='trapz32'):
        t = theta
        while (t < -np.pi):
            t = t+2.*np.pi
        while ( np.pi < t):
            t = t-2.*np.pi
        if (0. < t):
            intV, _ = mt.integrate(self.integrand, 0, t, args=(r), method=method)
            integral   = intV
        else:
            intV, _ = mt.integrate(self.integrand, t, 0., args=(r), method=method)
            integral   = -intV
        return phi - self.B0*self.R_axis*integral/self.dPsidr(r)
        
    def alpha0(self, r, theta, phi, method='trapz32'):
        '''Compute alpha(r, theta, phi=0) without the dPsidr division.'''
        t = theta
        while (t < -np.pi):
            t = t+2.*np.pi
        while ( np.pi < t):
            t = t-2.*np.pi
        if (0. < t):
            intV, _ = mt.integrate(self.integrand, 0, t, args=(r), method=method)
            integral   = intV
        else:
            intV, _ = mt.integrate(self.integrand, t, 0., args=(r), method=method)
            integral   = -intV
        return phi - self.B0*self.R_axis*integral
    
    def get_conf_grid(self):
        return [self.x, self.y, self.z]
    
    def get_toroidal_mode_number(self):
        self.n0 = 2.*np.pi*self.r0/self.qprofile_R(self.R0)/self.Ly # toroidal mode number
        return self.n0
    
    def get_epsilon(self):
        return self.a_mid/self.R_axis
    
    def qprofile_x(self,x):
        return self.qprofile_R(self.R_x(x))

    def info(self):
        print(f"R_axis: {self.R_axis}")
        print(f"Z_axis: {self.Z_axis}")
        print(f"R_LCFSmid: {self.R_LCFSmid}")
        print(f"B_axis: {self.B_axis}")
        print(f"a_shift: {self.a_shift}")
        print(f"q0: {self.q0}")
        print(f"kappa: {self.kappa}")
        print(f"delta: {self.delta}")
        print(f"x_LCFS: {self.x_LCFS}")
        print(f"geom_type: {self.geom_type}")
        print(f"Rmid_min: {self.Rmid_min}")
        print(f"Rmid_max: {self.Rmid_max}")
        print(f"R0: {self.R0}")
        print(f"r0: {self.r0}")
        print(f"B0: {self.B0}")
        print(f"a_mid: {self.a_mid}")

    def plot_qprofile(self,x=None):
        import matplotlib.pyplot as plt
        if x is None: x = self.x
        q = self.qprofile_x(x)
        plt.plot(self.R_x(x),q)
        # show domain limits
        plt.axvline(x=self.Rmid_min, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=self.Rmid_max, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('R [m]')
        plt.ylabel('q')
        plt.show()