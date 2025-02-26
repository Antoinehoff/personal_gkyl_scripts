local Gyrokinetic = G0.Gyrokinetic

-- Mathematical constants (dimensionless).
pi = math.pi

-- Physical constants (using non-normalized physical units).
epsilon0 = 8.854187817620389850536563031710750260608e-12 -- Permittivity of free space (F/m).
mass_elc = 9.10938215e-31 -- Electron mass (kg).
mass_pro = 1.672621637e-27 -- Proton mass (kg).
charge_elc = -1.602176487e-19 -- Electron charge (C).
charge_pro = 1.602176487e-19 -- Proton charge (C).
eV = 1.602176487e-19 -- Electron volt (J).

-- Ion parameter
AMU = 2.01410177811 -- atomic mass unit (deuterium).
charge_num = 1.0 -- charge number.
mass_ion = AMU_D * nass_pro -- ion mass (kg).
charge_ion = charge_num * charge_pro

-- Reference temperatures and density
n0 = 2.0e19 -- Reference number density (1 / m^3).
temp_elc = 100.0 * eV -- Electron temperature (J)
temp_ion = 100.0 * eV -- Ion temperature (J)

-- Magnetic geometry
B_axis = 1.4 -- Magnetic field axis (T).
R_axis = 0.8867856264 -- Manetic axis major radius (m).
Z_axis = 0.1414361745 -- Magnetic axis height (m).
a_shift = 0.5 -- Parameter in Shafranov shift.
R_LCFSmid = 1.0870056099999 -- Midplane LCFS radius (m).
x_inner = 0.04 -- Length inside the LCFS (m).
x_outer = 0.08 -- Length outside the LCFS (m).
kappa = 1.4 -- Elongation
delta = -0.38 -- Triangularity
Rmid_min = R_LCFSmid - x_inner -- Min midplane radius (m).
Rmid_max = R_LCFSmid - x_outer -- Max midplane radius (m).
R0 = 0.5*(Rmid_min+Rmid_max) -- Radius in the middle of the simulation box (m).
a0 = R_LCFSmid-R_axis -- Minor radius at outboard midplane (m).
r0 = R0 - R_axis -- Minor radius of the simulation box (m).
B0 = B_axis * (R_axis/R0) -- Magnetic field magnitude in the middle of sim box (T).

qprofile = function (r_) -- Safety factor profile.
  local a = [497.3420166252413, -1408.736172826569, 1331.4134861681464,
  -419.00692601227627]
  return a[1]*(r_+R_axis)^3 + a[2]*(r_+R_axis)^2 + a[3]*(r_+R_axis) + a[4]
end

r_x = function (x_) -- Major radius as a function of x.
  return x_ + a0 - x_inner
end

R_rtheta = function (r_, t_) -- Major radius as a function of r and theta.
  return R_axis - a_shift*r_*r_/(2.*R_axis) + r_*math.cos(t_ + math.asin(delta)*math.sin(t_))
end

Z_rtheta = function (r_, t_) -- Height as a function of r and theta.
  return Z_axis + kappa*r_*math.sin(t_)
end

dRdr = function (r_, t_) -- Derivative of major radius with respect to r.
  return -a_shift*r_/(R_axis) + math.cos(t_ + math.asin(delta)*math.sin(t_))
end

dRdtheta = function (r_, t_) -- Derivative of major radius with respect to theta.
  return -r_*math.sin(t_ + math.asin(delta)*math.sin(t_))
            *(1.0 + math.asin(delta)*math.cos(t_))
end

dZdr = function (r_, t_) -- Derivative of height with respect to r.
  return kappa*math.sin(t_)
end

dZdtheta = function (r_, t_) -- Derivative of height with respect to theta.
  return kappa*r_*math.cos(t_)
end

Jr = function (r_, t_) -- Jacobian determinant.
  return R_rtheta(r_, t_)*(dRdr(r_,t_)*dZdt(r_,t_)-dRdt(r_,t_)*dZdr(r_,t_))
end

nu_frac = 0.1 -- Collision frequency fraction.
log_lambda_elc = 6.6 - 0.5 * math.log(n0 / 1.0e20) + 1.5 * math.log(temp_elc / charge_ion) -- Electron Coulomb logarithm.
log_lambda_ion = 6.6 - 0.5 * math.log(n0 / 1.0e20) + 1.5 * math.log(temp_ion / charge_ion) -- Ion Coulomb logarithm.
nu_elc = nu_frac * log_lambda_elc * math.pow(charge_ion, 4.0) * n0 /
  (6.0 * math.sqrt(2.0) * math.pow(pi, 3.0 / 2.0) * math.pow(epsilon0, 2.0) * math.sqrt(mass_elc) * math.pow(temp_elc, 3.0 / 2.0)) -- Electron collision frequency.
nu_ion = nu_frac * log_lambda_ion * math.pow(charge_ion, 4.0) * n0 /
  (12.0 * math.pow(pi, 3.0 / 2.0) * math.pow(epsilon0, 2.0) * math.sqrt(mass_ion) * math.pow(temp_ion, 3.0 / 2.0)) -- Ion collision frequency.

vth_elc = math.sqrt(temp_elc / mass_elc) -- Electron thermal velocity (m/s).
vth_ion = math.sqrt(temp_ion / mass_ion) -- Ion thermal velocity (m/s).
c_s = math.sqrt(temp_elc / mass_ion) -- Sound speed (m/s).
omega_ci = math.abs(charge_ion * B0 / mass_ion) -- Ion cyclotron frequency (rad/s).
rho_s = vth_ion / omega_ci -- Ion sound gyroradius (m).
mu_elc = mass_elc * math.pow(4.0 * vth_elc, 2.0) / (2.0 * B0) -- Electron magnetic moment.
mu_ion = mass_ion * math.pow(4.0 * vth_ion, 2.0) / (2.0 * B0) -- Ion magnetic moment.

-- Numerical parameters.
poly_order = 1 -- Polynomial order.
basis_type = "serendipity" -- Basis function set.
time_stepper = "rk3" -- Time integrator.
cfl_frac = 1.0 -- CFL coefficient.
Nx = 48 -- Cell count (configuration space: x-direction).
Ny = 32 -- Cell count (configuration space: y-direction).
Nz = 16 -- Cell count (configuration space: z-direction).
Nvpar = 16 -- Cell count (velocity space: parallel velocity direction).
Nmu = 6 -- Cell count (velocity space: magnetic moment direction).

Lx = Rmid_max-Rmid_min -- Domain size (configuration space: x-direction).
Ly = 150 * rho_s -- Domain size (configuration space: y-direction).
Lz = 2.0 * pi - 1e-6 -- Domain size (configuration space: z-direction).

vpar_max_elc = 4.0 * vth_elc -- Domain boundary (electron velocity space: parallel velocity direction).
mu_max_elc = 0.5 * mu_elc -- Domain boundary (electron velocity space: magnetic moment direction).
vpar_max_ion = 4.0 * vth_ion -- Domain boundary (ion velocity space: parallel velocity direction).
mu_max_ion = 0.5 * mu_ion -- Domain boundary (ion velocity space: magnetic moment direction).

t_end = 4.0e-3 -- Final simulation time (s).
num_frames = 2000 -- Number of output frames.
field_energy_calcs = GKYL_MAX_INT -- Number of times to calculate field energy.
integrated_mom_calcs = GKYL_MAX_INT -- Number of times to calculate integrated moments.
dt_failure_tol = 1.0e-3 -- Minimum allowable fraction of initial time-step.
num_failures_max = 20 -- Maximum allowable number of consecutive small time-steps.

gyrokineticApp = Gyrokinetic.App.new {

  tEnd = t_end,
  nFrame = num_frames,
  fieldEnergyCalcs = field_energy_calcs,
  integratedMomentCalcs = integrated_mom_calcs,
  dtFailureTol = dt_failure_tol,
  numFailuresMax = num_failures_max,
  lower = { 0, -0.5 * Ly, -0.5 * Lz },
  upper = { Lx, 0.5 * Ly,  0.5 * Lz },
  cells = { Nx, Ny, Nz },
  cflFrac = cfl_frac,

  --basis = basis_type,
  polyOrder = poly_order,
  timeStepper = time_stepper,

  -- Decomposition for configuration space.
  decompCuts = { 1 }, -- Cuts in each coodinate direction (x-direction only).

  -- Boundary conditions for configuration space.
  periodicDirs = { 2 }, -- Periodic directions (y-direction only).

  geometry = {
    geometryID = G0.Geometry.MapC2P,
    world = { 0.0, 0.0, 0.0 },

    -- Computational coordinates (x, y, z) from physical coordinates (X, Y, Z).
    mapc2p = function (t, zc)
      local x, y, z = zc[1], zc[2], zc[3]
      local r = x + a0 - x_inner
      local q0 = qprofile(r0, R_axis)
      local R = R_axis - a_shift*r*r/(2.*R_axis) + r*cos(z + asin(delta)*sin(z)) -- Major radius.
      local Z = Z_axis + kappa*r*sin(z); -- Height.
      local phi = -q0/r0*y - alpha(r, z, 0, ctx) -- Toroidal angle.
      local X = R * math.cos(phi)
      local Y = R * math.sin(phi)
      local Z = y
      return X, Y, Z
    end,

    -- Magnetic field strength.
    bmagFunc = function (t, zc)
      local x = zc[1]

      return B0 * R / x
    end
  },

  -- Electrons.
  elc = Gyrokinetic.Species.new {
    charge = charge_elc, mass = mass_elc,
    
    -- Velocity space grid.
    lower = { -1.0, 0.0 },
    upper = { 1.0, 1.0 },
    cells = { Nvpar, Nmu },
    polarizationDensity = n0,

    mapc2p = {
      -- Rescaled electron velocity space coordinates (vpar, mu) from old velocity space coordinates (cpvar, cmu).
      mapping = function (t, vc)
        local cvpar, cmu = vc[1], vc[2]
    
        local vpar = 0.0
        local mu = 0.0
    
        if cvpar < 0.0 then
          vpar = -vpar_max_elc * (cvpar * cvpar)
        else
          vpar = vpar_max_elc * (cvpar * cvpar)
        end
        mu = mu_max_elc * (cmu * cmu)
    
        return vpar, mu
      end
    },

    -- Initial conditions.
    projection = {
      projectionID = G0.Projection.MaxwellianPrimitive,

      densityInit = function (t, xn)
        local x, z = xn[1], xn[3]

        local src_density = math.max(math.exp(-((x - xmu_src) * (x - xmu_src)) / ((2.0 * xsigma_src) * (2.0 * xsigma_src))), floor_src) * n_src
        local src_temp = 0.0
        local n = 0
      
        if x < xmu_src + 3.0 * xsigma_src then
          src_temp = T_src
        else
          src_temp = (3.0 / 8.0) * T_src
        end
      
        local c_s_src = math.sqrt((5.0 / 3.0) * src_temp / mass_ion)
        local n_peak = 4.0 * math.sqrt(5.0) / 3.0 / c_s_src * (0.125 * Lz) * src_density
      
        if math.abs(z) <= 0.25 * Lz then
          n = 0.5 * n_peak * (1.0 + math.sqrt(1.0 - (z / (0.25 * Lz)) * (z / (0.25 * Lz)))) -- Electron total number density (left).
        else
          n = 0.5 * n_peak -- Electron total number density (right).
        end
        
        return n
      end,
      temperatureInit = function (t, xn)
        local x = xn[1]

        local T = 0.0

        if x < xmu_src + 3.0 * xsigma_src then
          T = (5.0 / 4.0) * Te -- Electron isotropic temperature (left).
        else
          T = 0.5 * Te -- Electron isotropic temperature (right).
        end

        return T
      end,
      parallelVelocityInit = function (t, xn)
        return 0.0 -- Electron parallel velocity.
      end
    },

    source = {
      sourceID = G0.Source.Proj,
  
      numSources = 1,
      projections = {
        {
          projectionID = G0.Projection.MaxwellianPrimitive,

          densityInit = function (t, xn)
            local x, z = xn[1], xn[3]

            local n = 0.0

            if math.abs(z) < 0.25 * Lz then
              n = math.max(math.exp(-((x - xmu_src) * (x - xmu_src)) / ((2.0 * xsigma_src) * (2.0 * xsigma_src))),
                floor_src) * n_src -- Electron source total number density (left).
            else
              n = 1.0e-40 * n_src -- Electron source total number density (right).
            end

            return n
          end,
          temperatureInit = function (t, xn)
            local x = xn[1]

            local T = 0.0

            if x < xmu_src + 3.0 * xsigma_src then
              T = T_src -- Electron source isotropic temperature (left).
            else
              T = (3.0 / 8.0) * T_src -- Electron source isotropic temperature (right).
            end

            return T -- Electron source isotropic temperature.
          end,
          parallelVelocityInit = function (t, xn)
            return 0.0 -- Electron source parallel velocity.
          end
        }
      }
    },

    collisions = {
      collisionID = G0.Collisions.LBO,

      selfNu = function (t, xn)
        return nu_elc
      end,

      numCrossCollisions = 1,
      collideWith = { "ion" }
    },

    bcx = {
      lower = {
        type = G0.SpeciesBc.bcZeroFlux
      },
      upper = {
        type = G0.SpeciesBc.bcZeroFlux
      }
    },
    bcz = {
      lower = {
        type = G0.SpeciesBc.bcGkSheath
      },
      upper = {
        type = G0.SpeciesBc.bcGkSheath
      }
    },

    evolve = true, -- Evolve species?
    diagnostics = { "M0", "M1", "M2", "M2par", "M2perp" }
  },

  -- Ions.
  ion = Gyrokinetic.Species.new {
    charge = charge_ion, mass = mass_ion,
    
    -- Velocity space grid.
    lower = { -1.0, 0.0 },
    upper = { 1.0, 1.0 },
    cells = { Nvpar, Nmu },
    polarizationDensity = n0,

    mapc2p = {
      -- Rescaled ion velocity space coordinates (vpar, mu) from old velocity space coordinates (cpvar, cmu).
      mapping = function (t, vc)
        local cvpar, cmu = vc[1], vc[2]
    
        local vpar = 0.0
        local mu = 0.0
    
        if cvpar < 0.0 then
          vpar = -vpar_max_ion * (cvpar * cvpar)
        else
          vpar = vpar_max_ion * (cvpar * cvpar)
        end
        mu = mu_max_ion * (cmu * cmu)
    
        return vpar, mu
      end
    },

    -- Initial conditions.
    projection = {
      projectionID = G0.Projection.MaxwellianPrimitive,

      densityInit = function (t, xn)
        local x, z = xn[1], xn[3]

        local src_density = math.max(math.exp(-((x - xmu_src) * (x - xmu_src)) / ((2.0 * xsigma_src) * (2.0 * xsigma_src))), floor_src) * n_src
        local src_temp = 0.0
        local n = 0
      
        if x < xmu_src + 3.0 * xsigma_src then
          src_temp = T_src
        else
          src_temp = (3.0 / 8.0) * T_src
        end
      
        local c_s_src = math.sqrt((5.0 / 3.0) * src_temp / mass_ion)
        local n_peak = 4.0 * math.sqrt(5.0) / 3.0 / c_s_src * (0.125 * Lz) * src_density
      
        if math.abs(z) <= 0.25 * Lz then
          n = 0.5 * n_peak * (1.0 + math.sqrt(1.0 - (z / (0.25 * Lz)) * (z / (0.25 * Lz)))) -- Ion total number density (left).
        else
          n = 0.5 * n_peak -- Ion total number density (right).
        end
        
        return n
      end,
      temperatureInit = function (t, xn)
        local x = xn[1]

        local T = 0.0

        if x < xmu_src + 3.0 * xsigma_src then
          T = (5.0 / 4.0) * Ti -- Ion isotropic temperature (left).
        else
          T = 0.5 * Ti -- Ion isotropic temperature (right).
        end

        return T
      end,
      parallelVelocityInit = function (t, xn)
        return 0.0 -- Ion parallel velocity.
      end
    },

    source = {
      sourceID = G0.Source.Proj,
  
      numSources = 1,
      projections = {
        {
          projectionID = G0.Projection.MaxwellianPrimitive,

          densityInit = function (t, xn)
            local x, z = xn[1], xn[3]

            local n = 0.0

            if math.abs(z) < 0.25 * Lz then
              n = math.max(math.exp(-((x - xmu_src) * (x - xmu_src)) / ((2.0 * xsigma_src) * (2.0 * xsigma_src))),
                floor_src) * n_src -- Ion source total number density (left).
            else
              n = 1.0e-40 * n_src -- Ion source total number density (right).
            end

            return n
          end,
          temperatureInit = function (t, xn)
            local x = xn[1]

            local T = 0.0

            if x < xmu_src + 3.0 * xsigma_src then
              T = T_src -- Ion source isotropic temperature (left).
            else
              T = (3.0 / 8.0) * T_src -- Ion source isotropic temperature (right).
            end

            return T -- Ion source isotropic temperature.
          end,
          parallelVelocityInit = function (t, xn)
            return 0.0 -- Ion source parallel velocity.
          end
        }
      }
    },

    collisions = {
      collisionID = G0.Collisions.LBO,

      selfNu = function (t, xn)
        return nu_ion
      end,

      numCrossCollisions = 1,
      collideWith = { "elc" }
    },

    bcx = {
      lower = {
        type = G0.SpeciesBc.bcZeroFlux
      },
      upper = {
        type = G0.SpeciesBc.bcZeroFlux
      }
    },
    bcz = {
      lower = {
        type = G0.SpeciesBc.bcGkSheath
      },
      upper = {
        type = G0.SpeciesBc.bcGkSheath
      }
    },

    evolve = true, -- Evolve species?
    diagnostics = { "FourMoments" }
  },

  -- Field.
  field = Gyrokinetic.Field.new {
    femParBc = G0.ParProjBc.None,

    poissonBcs = {
      lowerType = {
        G0.PoissonBc.bcDirichlet,
        G0.PoissonBc.bcPeriodic
      },
      upperType = {
        G0.PoissonBc.bcDirichlet,
        G0.PoissonBc.bcPeriodic
      },
      lowerValue = {
        0.0
      },
      upperValue = {
        0.0
      }
    }
  }
}

gyrokineticApp:run()