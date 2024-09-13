class GeomParam:
    def __init__(self, R_axis, Z_axis, R_LCFSmid, a_shift, q0, kappa, delta):
        """
        Initialize the GeomParam class with the provided geometric parameters.

        :param R_axis: Major radius (m)
        :param a_shift: Shafranov shift parameter
        :param Z_axis: Vertical position of magnetic axis (m)
        :param a_mid: Mid-plane minor radius (m)
        :param kappa: Elongation factor (dimensionless)
        :param delta: Triangularity factor (dimensionless)
        :param q0: Safety factor at magnetic axis (dimensionless)
        :param R_LCFSmid: Radius of the Last Closed Flux Surface at mid-plane (m)
        """
        self.R_axis = R_axis
        self.a_shift = a_shift
        self.Z_axis = Z_axis
        self.kappa = kappa
        self.delta = delta
        self.q0 = q0
        self.R_LCFSmid = R_LCFSmid
        self.a_mid = R_LCFSmid-R_axis
