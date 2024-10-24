import postgkyl as pg
import numpy as np
import scipy.integrate as integrate

class GeomParam:
    def __init__(self, R_axis=0.0, Z_axis=0.0, R_LCFSmid=0.0, B0=1.4, x_out = 0.08,
                 a_shift=0.0, q0=1.6, kappa=1.0, delta=0.0, x_LCFS=0.0, geom_type='Miller'):
        self.R_axis     = R_axis
        self.a_shift    = a_shift
        self.Z_axis     = Z_axis
        self.kappa      = kappa
        self.delta      = delta
        self.q0         = q0
        self.x_LCFS     = x_LCFS
        self.R_LCFSmid  = R_LCFSmid
        self.B0         = B0
        self.Rmid_min   = R_LCFSmid-x_LCFS # Minimum midplane major radius of simulation box [m].
        self.Rmid_max   = R_LCFSmid+x_out  # Maximum midplane major radius of simulation box [m].
        self.R0         = 0.5*(self.Rmid_min+self.Rmid_max)  # Major radius of the simulation box [m].
        self.r0         = self.R0-R_axis          # Minor radius of the simulation box [m].
        self.geom_type  = geom_type
        # self.a_mid      = R_LCFSmid-R_axis
        self.a_mid      = \
            R_axis/a_shift - np.sqrt(R_axis*(R_axis - 2*a_shift*R_LCFSmid + 2*a_shift*R_axis))/a_shift
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
        
    def load_metric(self,fileprefix):
        #-- load B (bmag)
        fname = fileprefix+'-'+'bmag.gkyl'
        Gdata = pg.data.GData(fname)
        dg = pg.data.GInterpModal(Gdata,1,'ms')
        dg.interpolate(0,overwrite=True)
        self.bmag = Gdata.get_values()
        self.bmag = self.bmag[:,:,:,0]
        
        #-- load grid
        self.grids = [0.5*(g[1:]+g[:-1]) for g in Gdata.get_grid() if len(g) > 1]
        self.x = self.grids[0]; self.y = self.grids[1]; self.z = self.grids[2]
        self.Lx    = self.x[-1]-self.x[0]
        self.Ly    = self.y[-1]-self.y[0]
        self.Lz    = self.z[-1]-self.z[0]
        #self.toroidal_mn =  2.*np.pi*self.R_LCFSmid/self.q0/self.Ly
        #-- compute associated derivatives
        self.dBdx = np.gradient(self.bmag, self.grids[0], axis=0)  # Derivative w.r.t x
        self.dBdy = np.gradient(self.bmag, self.grids[1], axis=1)  # Derivative w.r.t y
        self.dBdz = np.gradient(self.bmag, self.grids[2], axis=2)  # Derivative w.r.t z

        #-- load g_ij (not useful yet)
        # fname = simulation.data_param.fileprefix+'-'+'g_ij.gkyl'

        #-- Load b_x, b_y, and b_z
        fname = fileprefix+'-'+'b_i.gkyl'
        self.b_i = []
        for i in range(3):
            Gdata = pg.data.GData(fname)
            dg = pg.data.GInterpModal(Gdata,1,'ms')
            dg.interpolate(i,overwrite=True)
            tmp_ = Gdata.get_values()
            self.b_i.append(tmp_[:,:,:,0])

        #-- Load Jacobian
        fname = fileprefix+'-'+'jacobgeo.gkyl'
        Gdata = pg.data.GData(fname)
        dg = pg.data.GInterpModal(Gdata,1,'ms')
        dg.interpolate(0,overwrite=True)
        self.Jacobian = Gdata.get_values()
        self.Jacobian = self.Jacobian[:,:,:,0]
        J_yz          = np.trapz(self.Jacobian,self.x,axis=0)
        J_z           = np.trapz(J_yz,self.y,axis=0)
        self.intJac   = np.trapz(J_z,self.z,axis=0)

        #.Minor radius as a function of x:
    def r_x(self,xIn):
        return self.Rmid_min+xIn-self.R_axis
    
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

    #.Magnetic safety factor profile.
    def qprofile(self,rIn):
        qa = [497.3420166252413,-1408.736172826569,1331.4134861681464,-419.00692601227627]
        return qa[0]*(rIn+self.R_axis)**3 + qa[1]*(rIn+self.R_axis)**2 + qa[2]*(rIn+self.R_axis) + qa[3]

    ## Geometric relations for Miller geometry
    def R_f(self, r, theta):
        return self.R_axis + r*np.cos(theta + np.arcsin(self.delta)*np.sin(theta))
    def Z_f(self, r, theta):
        return self.kappa*r*np.sin(theta)

    #.Minor radius as a function of x:
    def r_x(self,xIn):
        return self.a_mid + xIn

    #.Analytic derivatives.
    def R_f_r(self, r,theta): 
        return np.cos(theta + np.arcsin(self.delta)*np.sin(theta))
    def R_f_theta(self, r,theta): 
        return -r*(np.arcsin(self.delta)*np.cos(theta)+1.)*np.sin(np.arcsin(self.delta)*np.sin(theta)+theta)
    def Z_f_r(self, r,theta):
        return self.kappa*np.sin(theta)
    def Z_f_theta(self, r,theta):
        return self.kappa*r*np.cos(theta)
    
    def Jr_f(self, r, theta):
        return self.R_f(r,theta)*\
            ( self.R_f_r(r,theta)*self.Z_f_theta(r,theta)-self.Z_f_r(r,theta)*self.R_f_theta(r,theta) )

    def integrand(self, t, r):
        return self.Jr_f(r,t)/np.power(self.R_f(r,t),2)

    def dPsidr_f(self, r, theta):
        integral, _ = integrate.quad(self.integrand, 0., 2.*np.pi, args=(r), epsabs=1.e-8)
        return self.B0*self.R_axis/(2.*np.pi*self.qprofile(r))*integral

    def alpha_f(self, r, theta, phi):
        t = theta
        while (t < -np.pi):
            t = t+2.*np.pi

        while ( np.pi < t):
            t = t-2.*np.pi

        if (0. < t):
            intV, intE = integrate.quad(self.integrand, 0., t, args=(r), epsabs=1.e-8)
            integral   = intV
        else:
            intV, intE = integrate.quad(self.integrand, t, 0., args=(r), epsabs=1.e-8)
            integral   = -intV

        return phi - self.B0*self.R_axis*integral/self.dPsidr_f(r,theta)
    
    #.Function that wraps x to [xMin,xMax].
    def wrap(self, x, xMin, xMax):
        return (((x-xMin) % (xMax-xMin)) + (xMax-xMin)) % (xMax-xMin) + xMin

    def get_conf_grid(self):
        return [self.x, self.y, self.z]
    
    def get_toroidal_mode_number(self):
        self.n0 = 2.*np.pi*self.r0/self.qprofile(self.r0)/self.Ly # toroidal mode number
        return self.n0