import postgkyl as pg
import numpy as np
class GeomParam:
    def __init__(self, R_axis=0.0, Z_axis=0.0, R_LCFSmid=0.0, 
                 a_shift=0.0, q0=1.6, kappa=1.0, delta=0.0, x_LCFS=0.0):
        self.R_axis     = R_axis
        self.a_shift    = a_shift
        self.Z_axis     = Z_axis
        self.kappa      = kappa
        self.delta      = delta
        self.q0         = q0
        self.x_LCFS     = x_LCFS
        self.R_LCFSmid  = R_LCFSmid
        self.a_mid      = R_LCFSmid-R_axis
        self.g_ij       = None
        self.gij        = None
        self.bmag       = None
        self.b_i        = None
        self.grids      = None
        self.Jacobian   = None
        self.dBdx       = None
        self.dBdy       = None
        self.dBdz       = None
        self.bxgradBoB2 = None
        self.x          = None
        self.Lx         = None
        self.y          = None
        self.Ly         = None
        self.z          = None
        self.Lz         = None

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
        self.Lx    = self.grids[0][-1]-self.grids[0][0]
        self.Ly    = self.grids[1][-1]-self.grids[1][0]
        self.Lz    = self.grids[2][-1]-self.grids[2][0]
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

        #-- Compute Jacobian
        fname = fileprefix+'-'+'jacobgeo.gkyl'
        Gdata = pg.data.GData(fname)
        dg = pg.data.GInterpModal(Gdata,1,'ms')
        dg.interpolate(0,overwrite=True)
        self.Jacobian = Gdata.get_values()
        self.Jacobian = self.Jacobian[:,:,:,0]


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
        # Now divide the cross product by Bmag^2
        self.bxgradBoB2 /= self.Jacobian * self.bmag**2

    def GBflux_model(self,b=1.2):
        z = self.grids[2]
        return np.sin(z)*np.exp(-np.power(np.abs(z),1.5)/(2.*b))

    def set_domain(self,geom_type='Miller',vessel_corners=[[0.6,1.2],[-0.7,0.7]],Ntheta=128):
        if geom_type == 'Miller':
            ## Miller geometry model
            def RZ_rtheta(R_, theta, delta, kappa):
                r = R_ - self.R_axis
                R = (self.R_axis - self.a_shift * r**2 / (2.0 * self.R_axis) + 
                        r * np.cos(theta + np.arcsin(delta) * np.sin(theta)))
                Z = self.Z_axis + kappa*r*np.sin(theta)
                return [R,Z]
            
            Rmid_min = self.R_LCFSmid - self.x_LCFS
            Rmid_max = Rmid_min+self.Lx
            theta = np.linspace(-np.pi,+np.pi,Ntheta)
            self.RZ_min  = RZ_rtheta(Rmid_min,theta,self.delta,self.kappa)
            self.RZ_max  = RZ_rtheta(Rmid_max,theta,self.delta,self.kappa),
            self.RZ_lcfs = RZ_rtheta(self.R_LCFSmid,theta,self.delta,self.kappa)
            self.vessel_corners = vessel_corners           
        elif geom_type == 'efit':
            a = 0
            # one day... (09/20/2024)
