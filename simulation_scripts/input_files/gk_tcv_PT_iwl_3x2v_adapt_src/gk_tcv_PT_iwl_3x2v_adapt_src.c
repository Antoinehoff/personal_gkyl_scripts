#include <math.h>
#include <stdio.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_const.h>
#include <gkyl_eqn_type.h>
#include <gkyl_fem_parproj.h>
#include <gkyl_fem_poisson_bctype.h>
#include <gkyl_gyrokinetic.h>
#include <gkyl_null_comm.h>

#ifdef GKYL_HAVE_MPI
#include <mpi.h>
#include <gkyl_mpi_comm.h>
#ifdef GKYL_HAVE_NCCL
#include <gkyl_nccl_comm.h>
#endif
#endif

#include <gkyl_math.h>
#include <rt_arg_parse.h>

// Define the context of the simulation. This stores global parameters.
struct gk_app_ctx {
    int cdim, vdim;
    // Geometry and magnetic field parameters
    double a_shift, Z_axis, R_axis, R0, a_mid, x_inner, r0, B0, kappa, delta, q0, Bref, x_LCFS;
    // Plasma parameters
    int num_species;
    double me, qe, mi, qi, n0, Te0, Ti0;
    // Collision parameter
    double nuFrac;
    // Source parameters
    double num_sources;
    bool adapt_energy_srcCORE, adapt_particle_srcCORE; 
    double center_srcCORE[3], sigma_srcCORE[3];
    double energy_srcCORE, particle_srcCORE;
    double floor_srcCORE;
    bool adapt_energy_srcRECY, adapt_particle_srcRECY;
    double center_srcRECY[3], sigma_srcRECY[3];
    double energy_srcRECY, particle_srcRECY;
    double floor_srcRECY;
    // Grid parameters
    double Lx, Ly, Lz;
    double x_min, x_max, y_min, y_max, z_min, z_max;
    int num_cell_x, num_cell_y, num_cell_z, num_cell_vpar, num_cell_mu;
    int cells[GKYL_MAX_DIM];
    double vpar_max_elc, mu_max_elc, vpar_max_ion, mu_max_ion;
    // Simulation control parameters
    double final_time, write_phase_freq;
    int num_frames, int_diag_calc_num, num_failures_max;
    double dt_failure_tol;
};

// Function prototypes (defined at the end of the file)
static double r_x(double x, double a_mid, double x_inner);
static void zero_func(double t, const double *xn, double *fout, void *ctx);
static void density_init(double t, const double *xn, double *fout, void *ctx);
static void temp_elc(double t, const double *xn, double *fout, void *ctx);
static void temp_ion(double t, const double *xn, double *fout, void *ctx);
static void mapc2p(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx);
static void mapc2p_vel_elc(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx);
static void mapc2p_vel_ion(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx);
static void bmag_func(double t, const double *xn, double *bmag, void *ctx);
static void bc_shift_func_lo(double t, const double *xn, double *fout, void *ctx);
static void bc_shift_func_up(double t, const double *xn, double *fout, void *ctx);
static void write_data(struct gkyl_tm_trigger *iot_conf, struct gkyl_tm_trigger *iot_phase,
                       gkyl_gyrokinetic_app *app, double t_curr, bool force_write);
static void calc_integrated_diagnostics(struct gkyl_tm_trigger *iot, gkyl_gyrokinetic_app *app, double t_curr, bool force_write);

// Piecewise linear fit for q-profile TCV #65130 (PT)
// Context: a_shift=0.408, Z_axis=0.1414, R_axis=0.8727, B_axis=1.4,
//          R_LCFSmid=1.0969, x_inner=0.04, x_outer=0.08, Nx=24
//          Npieces=8, delta=0.2826
// Polynomial coefficients: [497.3420166252413, -1408.736172826569, 1331.4134861681464, -419.00692601227627]
static double qprofile(double R) {
 if (R < 1.0695104064577) return 22.303605465907 * R + -21.857287041786;
 if (R >= 1.0695104064577 && R < 1.0830075238591) return 27.36981322788 * R + -27.275648964493;
 if (R >= 1.0830075238591 && R < 1.0963994504733) return 32.938183319514 * R + -33.306235669365;
 if (R >= 1.0963994504733 && R < 1.1096861863002) return 38.996055244684 * R + -39.948083119158;
 if (R >= 1.1096861863002 && R < 1.12286773134) return 45.530867563663 * R + -47.199674079594;
 if (R >= 1.12286773134 && R < 1.1359440855925) return 52.530157893248 * R + -55.058951332965;
 if (R >= 1.1359440855925 && R < 1.1489152490578) return 59.981562906815 * R + -63.52333078748;
 if (R >= 1.1489152490578) return 67.872818334243 * R + -72.589714482263;
}

struct gk_app_ctx create_ctx(void)
{
  // TCV #65130 (NT) discharge parameters.
  double P_exp     = 0.3819e6;        // P_sol measured [W]
  double a_shift   = 0.4080;             // Parameter in Shafranov shift.
  double kappa     = 1.3951;             // Elongation (=1 for no elongation).
  double delta     = 0.2826;           // Triangularity (=0 for no triangularity).
  double R_axis    = 0.8727;    // Magnetic axis major radius [m].
  double R_LCFSmid = 1.0969; // Major radius of the LCFS at the outboard midplane [m].
  double Z_axis    = 0.1414;    // Magnetic axis height [m].
  double B_axis    = 1.4;             // Magnetic field at the magnetic axis [T].
  // Note: after this line, all the parameters are simulation design parameters.

  // Reference values.
  double Te0  = 100; // Ref. electron temperature [eV].
  double Ti0  = 100; // Ref. ion temperature [eV].
  double n0   = 2.0e19;   // Ref. density [1/m^3]
  double Bref = 1.129;   // Ref. magnetic field [T].

  // Numerical resolutions.
  int num_cell_x = 48;
  int num_cell_y = 32;
  int num_cell_z = 16;
  int num_cell_vpar = 12;
  int num_cell_mu = 8;
  double x_inner = 0.04; // Width of the closed flux surface region [m].
  double x_outer = 0.08; // Width of the SOL region [m].
  double nrhos_y = 150; // Number of rho_s0 in the y-direction, used to set the domain size along y.
  double nvth_vpar_e = 6.0; // sets vpar max e in terms of ref. elc th. vel.
  double nvth_vpar_i = 6.0; // sets vpar max i in terms of ref. ion th. vel.
  double nvth_mu_e = 1.5; // sets mu max e in terms of mi*pow(4*vte0,2)/(2*B0).
  double nvth_mu_i = 1.5; // sets mu max i in terms of mi*pow(4*vti0,2)/(2*B0).
  double nuFrac = 1.0; // Fraction of the Coulomb collision frequency to use in the simulation.

  // Time parameters
  double final_time = 2.e-3;
  int num_frames = 1000;
  double write_phase_freq = 0.2;
  int int_diag_calc_num = num_frames*100;
  double dt_failure_tol = 1.0e-3; // Minimum allowable fraction of initial time-step.
  int num_failures_max = 20; // Maximum allowable number of consecutive small time-steps.

//END OF THE USER PARAMETERS
  //-------------------------------------------------------------
  // JS_COPY_START
  // JS script will copy this line to the end of the file.

  // Simulation set up
  int cdim = 3, vdim = 2; // Dimensionality.
  // Universal constant parameters.
  double eps0 = GKYL_EPSILON0, eV = GKYL_ELEMENTARY_CHARGE;
  double mp = GKYL_PROTON_MASS, me = GKYL_ELECTRON_MASS;
  double qi = eV; // ion charge
  double qe = -eV; // electron charge
  Te0 = Te0*eV; Ti0 = Ti0*eV; // Convert temperature to Joules.
  
  // Geometry and magnetic field.
  double Rmid_min  = R_LCFSmid - x_inner;      // Minimum midplane major radius of simulation box [m].
  double Rmid_max  = R_LCFSmid + x_outer;      // Maximum midplane major radius of simulation box [m].
  double R0        = 0.5*(Rmid_min+Rmid_max);  // Major radius of the simulation box [m].
  double a_mid     = R_LCFSmid-R_axis;   // Minor radius at outboard midplane [m].
  // Redefine a_mid with Shafranov shift, to ensure LCFS radial location.
  a_mid = R_axis/a_shift - sqrt(R_axis*(R_axis - 2*a_shift*R_LCFSmid + 2*a_shift*R_axis))/a_shift;
  double r0        = R0-R_axis;           // Minor radius of the simulation box [m].
  double B0        = B_axis*(R_axis/R0);  // Magnetic field magnitude in the simulation box [T].

  // Plasma parameters. Chosen based on the value of a cubic sline
  // between the last TS data inside the LCFS and the probe data in
  // in the far SOL, near R=0.475 m.
  int num_species = 2; // Number of species in the simulation.
  double AMU = 2.01410177811;
  double mi  = mp*AMU;   // Deuterium ions.
  double vte0 = sqrt(Te0/me), vti0 = sqrt(Ti0/mi); // Thermal speeds.
  double c_s0 = sqrt(Te0/mi);
  double omega_ci0 = fabs(qi*B0/mi);
  double rho_s0 = c_s0/omega_ci0;

  // Configuration domain parameters 
  double Lx        = Rmid_max-Rmid_min;   // Domain size along x.
  double x_min     = 0.;
  double x_max     = Lx;
  double x_LCFS    = R_LCFSmid - Rmid_min; // Radial location of the last closed flux surface.
  double q0        = qprofile(R0);  // Safety factor in the center of domain.

  double Ly        = nrhos_y*rho_s0;           // Domain size along y.
  // Adjust the domain size along y to have integer toroidal mode number.
  // We need: 2*pi*Cy/Ly = integer (Cy = r0/q0)
  Ly = 2.*M_PI*r0/q0/round(2.*M_PI*r0/q0/Ly); 
  double y_min     = -Ly/2.;
  double y_max     =  Ly/2.;

  double vol_frac = 1.0/(2.*M_PI*r0/q0/Ly);

  double Lz        = 2.*M_PI-1e-10;       // Domain size along magnetic field.
  double z_min     = -Lz/2.;
  double z_max     =  Lz/2.;

  // Velocity box dimensions
  double vpar_max_elc = nvth_vpar_e*vte0;
  double mu_max_elc   = nvth_mu_e*me*pow(4*vte0,2)/(2*B0);
  double vpar_max_ion = nvth_vpar_i*vti0;
  double mu_max_ion   = nvth_mu_i*mi*pow(4*vti0,2)/(2*B0);

  // Source parameters
  double P_inj = P_exp * vol_frac / num_species; // Injection power normalized to the volume fraction and per species [W]
  double num_sources = 2; // We have two sources per species: a core and a recycling.
  // Core source parameters
  bool adapt_energy_srcCORE = true;
  bool adapt_particle_srcCORE = true;
  double energy_srcCORE = P_inj; // [W]
  double particle_srcCORE = 0.0; // [1/s]
  double center_srcCORE[3] = {x_min, 0.0, -Lz/4}; // This is the position of the ion source,
  double sigma_srcCORE[3] = {0.03*Lx, 0.0, Lz/6}; //  the electron source will be at +Lz/2.
  double floor_srcCORE = 1e-10;
  // Recycling source parameters (we do not use it here)
  bool adapt_energy_srcRECY = false;
  bool adapt_particle_srcRECY = true;
  double energy_srcRECY = 0.0; // [W]
  double particle_srcRECY = 0.0; // [1/s]
  double center_srcRECY[3] = {0.75*x_LCFS, 0.0, M_PI};
  double sigma_srcRECY[3] = {0.25*x_LCFS, 0.0, 0.05*Lz};
  double floor_srcRECY = 1e-10;

  struct gk_app_ctx ctx = {
    .cdim = cdim,
    .vdim = vdim,
    .a_shift = a_shift,
    .R_axis = R_axis,
    .R0     = R0    ,
    .a_mid  = a_mid ,
    .x_inner = x_inner,
    .r0     = r0    ,
    .B0     = B0    ,
    .kappa  = kappa ,
    .delta  = delta ,
    .q0     = q0    ,
    .Lx     = Lx    ,
    .Ly     = Ly    ,
    .Lz     = Lz    ,
    .x_min = x_min,  .x_max = x_max,
    .y_min = y_min,  .y_max = y_max,
    .z_min = z_min,  .z_max = z_max,
    .Bref = Bref,
    .x_LCFS = x_LCFS,
    .num_species = num_species,
    .me = me,  .qe = qe,
    .mi = mi,  .qi = qi,
    .n0 = n0,  .Te0 = Te0,  .Ti0 = Ti0,
    .nuFrac = nuFrac,
    .num_sources = num_sources,
    .adapt_energy_srcCORE = adapt_energy_srcCORE,
    .adapt_particle_srcCORE = adapt_particle_srcCORE,
    .center_srcCORE = {center_srcCORE[0], center_srcCORE[1], center_srcCORE[2]},
    .sigma_srcCORE = {sigma_srcCORE[0], sigma_srcCORE[1], sigma_srcCORE[2]},
    .energy_srcCORE = energy_srcCORE,  .particle_srcCORE = particle_srcCORE,
    .floor_srcCORE = floor_srcCORE,
    .adapt_energy_srcRECY = adapt_energy_srcRECY,
    .adapt_particle_srcRECY = adapt_particle_srcRECY,
    .center_srcRECY = {center_srcRECY[0], center_srcRECY[1], center_srcRECY[2]},
    .sigma_srcRECY = {sigma_srcRECY[0], sigma_srcRECY[1], sigma_srcRECY[2]},
    .energy_srcRECY = energy_srcRECY,  .particle_srcRECY = particle_srcRECY,
    .floor_srcRECY = floor_srcRECY,
    .num_cell_x     = num_cell_x,
    .num_cell_y     = num_cell_y,
    .num_cell_z     = num_cell_z,
    .num_cell_vpar  = num_cell_vpar,
    .num_cell_mu    = num_cell_mu,
    .cells = {num_cell_x, num_cell_y, num_cell_z, num_cell_vpar, num_cell_mu},
    .vpar_max_elc = vpar_max_elc,  .mu_max_elc = mu_max_elc,
    .vpar_max_ion = vpar_max_ion,  .mu_max_ion = mu_max_ion,
    .write_phase_freq = write_phase_freq,
    .final_time = final_time,  .num_frames = num_frames,
    .int_diag_calc_num = int_diag_calc_num,
    .dt_failure_tol = dt_failure_tol,
    .num_failures_max = num_failures_max,
  };
  return ctx;
}

int 
main(int argc, char **argv)
{
  struct timespec timer_global = gkyl_wall_clock();
  struct gkyl_app_args app_args = parse_app_args(argc, argv);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
    MPI_Init(&argc, &argv);
#endif
  
  if (app_args.trace_mem) {
    gkyl_cu_dev_mem_debug_set(true);
    gkyl_mem_debug_set(true);
  }

  struct gk_app_ctx ctx = create_ctx(); // context for init functions

  // Verify that the number of z-plane is divisible by the number of cuts
  if (ctx.num_cell_z % app_args.cuts[ctx.cdim-1] != 0) {
      fprintf(stderr, "Error: Number of z-cells (%d) must be divisible by the number of cuts in z (%d)\n",
              ctx.num_cell_z, app_args.cuts[2]);
      return -1;
  }

  double max_run_time;
  sscanf(app_args.opt_args, "max_run_time=%lf", &max_run_time);
  int cells_x[ctx.cdim], cells_v[ctx.vdim];
  for (int d=0; d<ctx.cdim; d++)
    cells_x[d] = APP_ARGS_CHOOSE(app_args.xcells[d], ctx.cells[d]);
  for (int d=0; d<ctx.vdim; d++)
    cells_v[d] = APP_ARGS_CHOOSE(app_args.vcells[d], ctx.cells[ctx.cdim+d]);
  // Construct communicator for use in app.
  struct gkyl_comm *comm = gkyl_gyrokinetic_comms_new(app_args.use_mpi, app_args.use_gpu, stderr);
  int my_rank = 0;
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
    gkyl_comm_get_rank(comm, &my_rank);
#endif


  struct gkyl_gyrokinetic_projection proj_srcCORE_e = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN,
    .gaussian_mean = {ctx.center_srcCORE[0], ctx.center_srcCORE[1], -ctx.center_srcCORE[2]},
    .gaussian_std_dev = {ctx.sigma_srcCORE[0], ctx.sigma_srcCORE[1], ctx.sigma_srcCORE[2]},
    .total_num_particles = ctx.particle_srcCORE,
    .total_kin_energy = ctx.energy_srcCORE,
    .temp_max = 5.0*ctx.Te0,
    .temp_min = 0.1*ctx.Te0,
    .f_floor = ctx.floor_srcCORE,
  };

  struct gkyl_gyrokinetic_adapt_source adapt_srcCORE_e ={
    .adapt_to_species = "elc",
    .adapt_particle = ctx.adapt_particle_srcCORE,
    .adapt_energy = ctx.adapt_energy_srcCORE,
    .num_boundaries = 1,
    .dir = {0, 0, 2, 2},
    .edge = {GKYL_LOWER_EDGE, GKYL_UPPER_EDGE, GKYL_LOWER_EDGE, GKYL_UPPER_EDGE},
  };

  struct gkyl_gyrokinetic_projection proj_srcRECY_e = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN,
    .gaussian_mean = {ctx.center_srcRECY[0], ctx.center_srcRECY[1], ctx.center_srcRECY[2]},
    .gaussian_std_dev = {ctx.sigma_srcRECY[0], ctx.sigma_srcRECY[1], ctx.sigma_srcRECY[2]},
    .total_num_particles = ctx.particle_srcRECY,
    .total_kin_energy = ctx.energy_srcRECY,
    .temp_max = 5.0*ctx.Te0,
    .temp_min = 0.1*ctx.Te0,
    .f_floor = ctx.floor_srcRECY,
  };

  struct gkyl_gyrokinetic_adapt_source adapt_srcRECY_e = {
    .adapt_to_species = "ion",
    .adapt_particle = ctx.adapt_particle_srcRECY,
    .adapt_energy = ctx.adapt_energy_srcRECY,
    .num_boundaries = 3,
    .dir = {0, 2, 2},
    .edge = {GKYL_UPPER_EDGE, GKYL_LOWER_EDGE, GKYL_UPPER_EDGE},
  };

  // electrons
  struct gkyl_gyrokinetic_species elc = {
    .name = "elc",
    .charge = ctx.qe, .mass = ctx.me,
    .lower = { -1.0/sqrt(2.0), 0.0},
    .upper = {  1.0/sqrt(2.0), 1.0},
    .cells = { cells_v[0], cells_v[1] },
    .polarization_density = ctx.n0,

/*
    .init_from_file = {
       .type = GKYL_IC_IMPORT_F,
       .file_name = "restart-elc.gkyl"
    },
//*/  
    .mapc2p = {
      .mapping = mapc2p_vel_elc,
      .ctx = &ctx,
    },

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .ctx_upar = &ctx,
      .ctx_temp = &ctx,
      .density = density_init,
      .upar = zero_func,
      .temp = temp_elc,
    },

    .collisions =  {
      .collision_id = GKYL_LBO_COLLISIONS,
      .normNu = true,
      .nuFrac = ctx.nuFrac,
      .n_ref = ctx.n0, // Density used to calculate coulomb logarithm
      .T_ref = ctx.Te0, // Temperature used to calculate coulomb logarithm
      .ctx = &ctx,
      .num_cross_collisions = 1,
      .collide_with = { "ion"},
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = ctx.num_sources,
      .num_adapt_sources = ctx.num_sources,
      .projection[0] = proj_srcCORE_e,
      .adapt[0] = adapt_srcCORE_e,
      .projection[1] = proj_srcRECY_e,
      .adapt[1] = adapt_srcRECY_e,
      .diagnostics = {
        .num_diag_moments = 1,
        .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
        .num_integrated_diag_moments = 1,
        .integrated_diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
      }
    },

    .bcx = {
      .lower={.type = GKYL_SPECIES_ABSORB,},
      .upper={.type = GKYL_SPECIES_ABSORB,},
    },
    .bcz = {
      .lower={.type = GKYL_SPECIES_GK_IWL,
              .aux_profile = bc_shift_func_lo,
              .aux_ctx = &ctx,
      },
      .upper={.type = GKYL_SPECIES_GK_IWL,
              .aux_profile = bc_shift_func_up,
              .aux_ctx = &ctx,
      },
    },
    .num_diag_moments = 9,
    .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN, GKYL_F_MOMENT_BIMAXWELLIAN, 
      GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2PAR, GKYL_F_MOMENT_M2PERP, 
      GKYL_F_MOMENT_M2, GKYL_F_MOMENT_M3PAR, GKYL_F_MOMENT_M3PERP},
    .num_integrated_diag_moments = 1,
    .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    .boundary_flux_diagnostics = {
      .num_integrated_diag_moments = 1,
      .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    },
  };

  // ions

  struct gkyl_gyrokinetic_projection proj_srcCORE_i = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN,
    .gaussian_mean = {ctx.center_srcCORE[0], ctx.center_srcCORE[1], ctx.center_srcCORE[2]},
    .gaussian_std_dev = {ctx.sigma_srcCORE[0], ctx.sigma_srcCORE[1], ctx.sigma_srcCORE[2]},
    .total_num_particles = ctx.particle_srcCORE,
    .total_kin_energy = ctx.energy_srcCORE,
    .temp_max = 5.0*ctx.Te0,
    .temp_min = 0.1*ctx.Te0,
    .f_floor = ctx.floor_srcCORE,
  };

  struct gkyl_gyrokinetic_adapt_source adapt_srcCORE_i ={
    .adapt_to_species = "ion",
    .adapt_particle = ctx.adapt_particle_srcCORE,
    .adapt_energy = ctx.adapt_energy_srcCORE,
    .num_boundaries = 1,
    .dir = {0, 0, 2, 2},
    .edge = {GKYL_LOWER_EDGE, GKYL_UPPER_EDGE, GKYL_LOWER_EDGE, GKYL_UPPER_EDGE},
  };

  struct gkyl_gyrokinetic_projection proj_srcRECY_i = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN,
    .gaussian_mean = {ctx.center_srcRECY[0], ctx.center_srcRECY[1], ctx.center_srcRECY[2]},
    .gaussian_std_dev = {ctx.sigma_srcRECY[0], ctx.sigma_srcRECY[1], ctx.sigma_srcRECY[2]},
    .total_num_particles = ctx.particle_srcRECY,
    .total_kin_energy = ctx.energy_srcRECY,
    .temp_max = 5.0*ctx.Te0,
    .temp_min = 0.1*ctx.Te0,
    .f_floor = ctx.floor_srcRECY,
  };

  struct gkyl_gyrokinetic_adapt_source adapt_srcRECY_i = {
    .adapt_to_species = "ion",
    .adapt_particle = ctx.adapt_particle_srcRECY,
    .adapt_energy = ctx.adapt_energy_srcRECY,
    .num_boundaries = 3,
    .dir = {0, 2, 2},
    .edge = {GKYL_UPPER_EDGE, GKYL_LOWER_EDGE, GKYL_UPPER_EDGE},
  };

  struct gkyl_gyrokinetic_species ion = {
    .name = "ion",
    .charge = ctx.qi, .mass = ctx.mi,
    .lower = { -1.0/sqrt(2.0), 0.0},
    .upper = {  1.0/sqrt(2.0), 1.0},
    .cells = { cells_v[0], cells_v[1] },
    .polarization_density = ctx.n0,

/*
    .init_from_file = {
       .type = GKYL_IC_IMPORT_F,
       .file_name = "restart-ion.gkyl"
    },
//*/
    .mapc2p = {
      .mapping = mapc2p_vel_ion,
      .ctx = &ctx,
    },

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .ctx_upar = &ctx,
      .ctx_temp = &ctx,
      .density = density_init,
      .upar = zero_func,
      .temp = temp_ion,
    },

    .collisions =  {
      .collision_id = GKYL_LBO_COLLISIONS,
      .normNu = true,
      .nuFrac = ctx.nuFrac,
      .n_ref = ctx.n0, // Density used to calculate coulomb logarithm
      .T_ref = ctx.Ti0, // Temperature used to calculate coulomb logarithm
      .ctx = &ctx,
      .num_cross_collisions = 1,
      .collide_with = { "elc"},
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = ctx.num_sources,
      .num_adapt_sources = ctx.num_sources,
      .projection[0] = proj_srcCORE_i,
      .adapt[0] = adapt_srcCORE_i,
      .projection[1] = proj_srcRECY_i,
      .adapt[1] = adapt_srcRECY_i,
      .diagnostics = {
        .num_diag_moments = 1,
        .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
        .num_integrated_diag_moments = 1,
        .integrated_diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
      }
    },
    .bcx = {
      .lower={.type = GKYL_SPECIES_ABSORB,},
      .upper={.type = GKYL_SPECIES_ABSORB,},
    },
    .bcz = {
      .lower={.type = GKYL_SPECIES_GK_IWL,
              .aux_profile = bc_shift_func_lo,
              .aux_ctx = &ctx,
      },
      .upper={.type = GKYL_SPECIES_GK_IWL,
              .aux_profile = bc_shift_func_up,
              .aux_ctx = &ctx,
      },
    },
    .num_diag_moments = 9,
    .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN, GKYL_F_MOMENT_BIMAXWELLIAN, 
      GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2PAR, GKYL_F_MOMENT_M2PERP, 
      GKYL_F_MOMENT_M2, GKYL_F_MOMENT_M3PAR, GKYL_F_MOMENT_M3PERP},
    .num_integrated_diag_moments = 1,
    .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    .boundary_flux_diagnostics = {
      .num_integrated_diag_moments = 1,
      .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    },
  };

  struct gkyl_poisson_bias_plane target_corner_bc = {
    .dir = 0, // Direction perpendicular to the plane.
    .loc = ctx.x_LCFS, // Location of the plane in the 'dir' dimension.
    .val = 0.0, // Biasing value.
  };
  
  struct gkyl_poisson_bias_plane_list bias_plane_list = {
    .num_bias_plane = 1,
    .bp = &target_corner_bc,
  };

  // field
  struct gkyl_gyrokinetic_field field = {
    .gkfield_id = GKYL_GK_FIELD_ES_IWL,
    .polarization_bmag = ctx.Bref,
    .poisson_bcs = {.lo_type = {GKYL_POISSON_DIRICHLET},
                    .up_type = {GKYL_POISSON_DIRICHLET},
                    .lo_value = {0.0}, .up_value = {0.0},
                   },
    .bias_plane_list = &bias_plane_list,
    .time_rate_diagnostics = true,
  };

  // Geometry
  struct gkyl_gyrokinetic_geometry geometry = {
    .geometry_id = GKYL_MAPC2P,
    .world = {0.},
    .mapc2p = mapc2p, // mapping of cCOREutational to physical space
    .c2p_ctx = &ctx,
    .bmag_func = bmag_func, // magnetic field magnitude
    .bmag_ctx = &ctx,
    .has_LCFS = true,
    .x_LCFS = ctx.x_LCFS,
  };

  // Parallelism
  struct gkyl_app_parallelism_inp parallelism = {
    .comm = comm,
    .cuts = {app_args.cuts[0], app_args.cuts[1], app_args.cuts[2]},
    .use_gpu = app_args.use_gpu,
  };

  // GK app
  struct gkyl_gk *gk = gkyl_malloc(sizeof *gk);
  memset(gk, 0, sizeof(*gk));

  strcpy(gk->name, "rt_gk_tcv_nt_iwl_adapt_src_3x2v_p1");

  gk->cfl_frac_omegaH = 1.0e9;
  gk->cfl_frac = 1.0;
  gk->cdim = 3;
  gk->vdim = 2;
  gk->lower[0] = ctx.x_min;
  gk->lower[1] = ctx.y_min;
  gk->lower[2] = ctx.z_min;
  gk->upper[0] = ctx.x_max;
  gk->upper[1] = ctx.y_max;
  gk->upper[2] = ctx.z_max;
  gk->cells[0] = cells_x[0];
  gk->cells[1] = cells_x[1];
  gk->cells[2] = cells_x[2];
  gk->poly_order = 1;
  gk->basis_type = app_args.basis_type;
  gk->geometry = geometry;
  gk->num_periodic_dir = 1;
  gk->periodic_dirs[0] = 1;
  gk->num_species = ctx.num_species;
  gk->species[0] = elc;
  gk->species[1] = ion;
  gk->field = field;
  gk->parallelism = parallelism;

  // create app object
  if (my_rank == 0) printf("Creating app object ...\n");
  gkyl_gyrokinetic_app *app = gkyl_gyrokinetic_app_new(gk);
  
  // Initial and final simulation times.
  int frame_curr = 0;
  double t_curr = 0.0, t_end = ctx.final_time;
  // Initialize simulation.
  if (app_args.is_restart) {
    struct gkyl_app_restart_status status = gkyl_gyrokinetic_app_read_from_frame(app, app_args.restart_frame);

    if (status.io_status != GKYL_ARRAY_RIO_SUCCESS) {
      gkyl_gyrokinetic_app_cout(app, stderr, "*** Failed to read restart file! (%s)\n",
        gkyl_array_rio_status_msg(status.io_status));
      goto freeresources;
    }

    frame_curr = status.frame;
    t_curr = status.stime;

    gkyl_gyrokinetic_app_cout(app, stdout, "Restarting from frame %d", frame_curr);
    gkyl_gyrokinetic_app_cout(app, stdout, " at time = %g\n", t_curr);
  }
  else {
    gkyl_gyrokinetic_app_apply_ic(app, t_curr);
  }  

  // Create triggers for IO.
  int num_frames = ctx.num_frames, num_int_diag_calc = ctx.int_diag_calc_num;
  struct gkyl_tm_trigger trig_write_conf = { .dt = t_end/num_frames, .tcurr = t_curr, .curr = frame_curr };
  struct gkyl_tm_trigger trig_write_phase = { .dt = t_end/(ctx.write_phase_freq*num_frames), .tcurr = t_curr, .curr = frame_curr};
  struct gkyl_tm_trigger trig_calc_intdiag = { .dt = t_end/GKYL_MAX2(num_frames, num_int_diag_calc),
    .tcurr = t_curr, .curr = frame_curr };

  // Write out ICs (if restart, it overwrites the restart frame).
  if (!app_args.is_restart) {
    calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, false);
    write_data(&trig_write_conf, &trig_write_phase, app, t_curr, false);
  }

  // Initial time-step.
  double dt = t_end-t_curr;

  // Initialize small time-step check.
  double dt_init = -1.0, dt_failure_tol = ctx.dt_failure_tol;
  int num_failures = 0, num_failures_max = ctx.num_failures_max;

  // MAIN TIME LOOP
  if (my_rank == 0) printf("Starting main loop ...\n");
  long step = 1, num_steps = app_args.num_steps;
  while ((t_curr < t_end) && (step <= num_steps)) {
    if (step == 1 || step % 100 == 0)
      gkyl_gyrokinetic_app_cout(app, stdout, "Taking time-step at t = %g mus ...", t_curr*1.0e6);
    struct gkyl_update_status status = gkyl_gyrokinetic_update(app, dt);
    if (step == 1 || step % 100 == 0)
      gkyl_gyrokinetic_app_cout(app, stdout, " dt = %g mus\n", status.dt_actual*1.0e6);

    if (!status.success) {
      if (my_rank == 0) gkyl_gyrokinetic_app_cout(app, stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }
    t_curr += status.dt_actual;
    dt = status.dt_suggested;

    calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, t_curr > t_end);
    write_data(&trig_write_conf, &trig_write_phase, app, t_curr, t_curr > t_end);

    if (dt_init < 0.0) {
      dt_init = status.dt_actual;
    }
    else if (status.dt_actual < dt_failure_tol * dt_init) {
      num_failures += 1;

      gkyl_gyrokinetic_app_cout(app, stdout, "WARNING: Time-step dt = %g", status.dt_actual);
      gkyl_gyrokinetic_app_cout(app, stdout, " is below %g*dt_init ...", dt_failure_tol);
      gkyl_gyrokinetic_app_cout(app, stdout, " num_failures = %d\n", num_failures);
      if (num_failures >= num_failures_max) {
        gkyl_gyrokinetic_app_cout(app, stdout, "ERROR: Time-step was below %g*dt_init ", dt_failure_tol);
        gkyl_gyrokinetic_app_cout(app, stdout, "%d consecutive times. Aborting simulation ....\n", num_failures_max);
        calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, true);
        write_data(&trig_write_conf, &trig_write_phase, app, t_curr, true);
        break;
      }
    }
    else {
      num_failures = 0;
    }

    step += 1;

    if (max_run_time > 0.0 && 
        gkyl_time_diff_now_sec(timer_global) > max_run_time) {
      if (my_rank == 0) gkyl_gyrokinetic_app_cout(app, stdout, "Reached max run time, exiting ...\n");
      break;
    }
  }
  if (my_rank == 0) printf(" ... finished\n");
  gkyl_gyrokinetic_app_stat_write(app);
  
  // fetch simulation statistics
  struct gkyl_gyrokinetic_stat stat = gkyl_gyrokinetic_app_stat(app);

  gkyl_gyrokinetic_app_cout(app, stdout, "\n");
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of update calls %ld\n", stat.nup);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of forward-Euler calls %ld\n", stat.nfeuler);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of RK stage-2 failures %ld\n", stat.nstage_2_fail);
  if (stat.nstage_2_fail > 0) {
    gkyl_gyrokinetic_app_cout(app, stdout, "Max rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[1]);
    gkyl_gyrokinetic_app_cout(app, stdout, "Min rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[0]);
  }  
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of RK stage-3 failures %ld\n", stat.nstage_3_fail);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of write calls %ld\n", stat.n_io);
  gkyl_gyrokinetic_app_print_timings(app, stdout);

  freeresources:
  // simulation cCORElete, free app
  gkyl_gyrokinetic_app_release(app);
  gkyl_gyrokinetic_comms_release(comm);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Finalize();
  }
#endif
  
  return 0;
}

// End of main function
// ---------------------------------------------------------------

// Function definitions
// ---------------------------------------------------------------

// Geometry related functions 
double r_x(double x, double a_mid, double x_inner)
{
  return x+a_mid-x_inner;
}

double R_rtheta(double r, double theta, void *ctx)
{
  // Major radius as a function of minor radius r and poloidal angle theta.
  struct gk_app_ctx *app = ctx;
  double a_shift = app->a_shift;
  double R_axis = app->R_axis;
  double delta = app->delta;
  return R_axis - a_shift*r*r/(2.*R_axis) + r*cos(theta + asin(delta)*sin(theta));
}

double Z_rtheta(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double Z_axis = app->Z_axis;
  double kappa = app->kappa;
  return Z_axis + kappa*r*sin(theta);
}

double dRdr(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double a_shift = app->a_shift;
  double R_axis = app->R_axis;
  double delta = app->delta;
  return - a_shift*r/(R_axis) + cos(theta + asin(delta)*sin(theta));
}

double dRdtheta(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double delta = app->delta;
  return -r*sin(theta + asin(delta)*sin(theta))*(1.+asin(delta)*cos(theta));
}

double dZdr(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double kappa = app->kappa;
  return kappa*sin(theta);
}

double dZdtheta(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double kappa = app->kappa;
  return kappa*r*cos(theta);
}

double Jr(double r, double theta, void *ctx)
{
  return R_rtheta(r,theta,ctx)*( dRdr(r,theta,ctx) * dZdtheta(r,theta,ctx)
		                -dRdtheta(r,theta,ctx) * dZdr(r,theta,ctx) );
}

struct integrand_ctx {
  struct gk_app_ctx *app_ctx;
  double r;
};

double integrand(double t, void *int_ctx)
{
  struct integrand_ctx *inctx = int_ctx;
  double r = inctx->r;
  struct gk_app_ctx *app = inctx->app_ctx;
  return Jr(r,t,app) / pow(R_rtheta(r,t,app),2);
}

double dPsidr(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  struct integrand_ctx tmp_ctx = {.app_ctx = app, .r = r};
  struct gkyl_qr_res integral;
  integral = gkyl_dbl_exp(integrand, &tmp_ctx, 0., 2.*M_PI, 7, 1e-10);

  double B0 = app->B0;
  double a_mid = app->a_mid;
  double R_axis = app->R_axis;
  double R = R_rtheta(r, 0.0, ctx);
  return ( B0*R_axis/(2.*M_PI*qprofile(R)))*integral.res;
}

double alpha(double r, double theta, double phi, void *ctx)
{
  double twrap = theta;
  while (twrap < -M_PI) twrap = twrap+2.*M_PI;
  while (M_PI < twrap) twrap = twrap-2.*M_PI;

  struct gk_app_ctx *app = ctx;
  struct integrand_ctx tmp_ctx = {.app_ctx = app, .r = r};
  struct gkyl_qr_res integral;
  if (0. < twrap) {
    integral = gkyl_dbl_exp(integrand, &tmp_ctx, 0., twrap, 7, 1e-10);
  } else {
    integral = gkyl_dbl_exp(integrand, &tmp_ctx, twrap, 0., 7, 1e-10);
    integral.res = -integral.res;
  }

  double B0 = app->B0;
  double R_axis = app->R_axis;

  return phi - B0*R_axis*integral.res/dPsidr(r,theta,ctx);
}

double Bphi(double R, void *ctx)
{
  // Toroidal magnetic field.
  struct gk_app_ctx *app = ctx;
  double B0 = app->B0;
  double R0 = app->R0;
  return B0*R0/R;
}

double gradr(double r, double theta, void *ctx)
{
  return (R_rtheta(r,theta,ctx)/Jr(r,theta,ctx))*sqrt(pow(dRdtheta(r,theta,ctx),2) + pow(dZdtheta(r,theta,ctx),2));
}

void zero_func(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  fout[0] = 0.0;
}

// Density initial condition (like TCV exp profile)
void density_init(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];
  struct gk_app_ctx *app = ctx;
  double n0 = 5e19;
  double x0 = -0.03;
  double c1 = 0.5;
  double c2 = 8.0;
  double c3 = 0.005;
  fout[0] = n0*(c1*(1.+tanh(c2*(-10*(x+x0))))+c3);
}

// Electron temperature initial conditions
void temp_elc(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];
  struct gk_app_ctx *app = ctx;
  double T0 = 200 * GKYL_ELEMENTARY_CHARGE;
  double x0 = -0.03; // position of the transition region
  double c0 = 1.3; // multiplicative factor
  double c1 = 0.5; // control the temperature at the _core
  double c2 = 8.0; // control the width of the transition region
  double c3 = 0.1; // control the temperature at the SOL
  fout[0] = c0*T0*(c1*(1.+tanh(c2*(-10*(x+x0))))+c3);
}

// Ion temperature initial conditions
void temp_ion(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];
  struct gk_app_ctx *app = ctx;
  double T0 = 200 * GKYL_ELEMENTARY_CHARGE;
  double x0 = -0.04; // position of the transition region
  double c0 = 1.0; // multiplicative factor
  double c1 = 0.5; // control the temperature at the _core
  double c2 = 3.0; // control the width of the transition region
  double c3 = 0.2; // control the temperature at the SOL
  fout[0] = c0*T0*(c1*(1.+tanh(c2*(-10*(x+x0))))+c3);
}

// Geometry evaluation functions for the gk app
void mapc2p(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx)
{
  double x = xc[0], y = xc[1], z = xc[2];
  struct gk_app_ctx *app = ctx;
  double r0 = app->r0;
  double q0 = app->q0;
  double a_mid = app->a_mid;
  double x_inner = app->x_inner;
  double r = r_x(x,a_mid,x_inner);
  // Map to cylindrical (R, Z, phi) coordinates.
  double R   = R_rtheta(r, z, ctx);
  double Z   = Z_rtheta(r, z, ctx);
  double phi = -q0/r0*y - alpha(r, z, 0, ctx);
  // Map to Cartesian (X, Y, Z) coordinates.
  double X = R*cos(phi);
  double Y = R*sin(phi);
  xp[0] = X; xp[1] = Y; xp[2] = Z;
}

// Taken from rt gk d3d 3x2c, is this the non uniform v grid mapping?
void mapc2p_vel_elc(double t, const double *vc, double* GKYL_RESTRICT vp, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double vpar_max_elc = app->vpar_max_elc;
  double mu_max_elc = app->mu_max_elc;
  double cvpar = vc[0], cmu = vc[1];
  // Linear map up to vpar_max/2, then quadratic.
  if (fabs(cvpar) <= 0.5)
    vp[0] = vpar_max_elc*cvpar;
  else if (cvpar < -0.5)
    vp[0] = -vpar_max_elc*2.0*pow(cvpar,2);
  else
    vp[0] =  vpar_max_elc*2.0*pow(cvpar,2);
  // Quadratic map in mu.
  vp[1] = mu_max_elc*pow(cmu,2);
}

void mapc2p_vel_ion(double t, const double *vc, double* GKYL_RESTRICT vp, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double vpar_max_ion = app->vpar_max_ion;
  double mu_max_ion = app->mu_max_ion;
  double cvpar = vc[0], cmu = vc[1];
  // Linear map up to vpar_max/2, then quadratic.
  if (fabs(cvpar) <= 0.5)
    vp[0] = vpar_max_ion*cvpar;
  else if (cvpar < -0.5)
    vp[0] = -vpar_max_ion*2.0*pow(cvpar,2);
  else
    vp[0] =  vpar_max_ion*2.0*pow(cvpar,2);
  // Quadratic map in mu.
  vp[1] = mu_max_ion*pow(cmu,2);
}

void bmag_func(double t, const double *xc, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xc[0], y = xc[1], z = xc[2];
  struct gk_app_ctx *app = ctx;
  double a_mid = app->a_mid;
  double x_inner = app->x_inner;
  double r = r_x(x,a_mid,x_inner);
  double Bt = Bphi(R_rtheta(r,z,ctx),ctx);
  double Bp = dPsidr(r,z,ctx)/R_rtheta(r,z,ctx)*gradr(r,z,ctx);
  fout[0] = sqrt(Bt*Bt + Bp*Bp);
}

void bc_shift_func_lo(double t, const double *xc, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xc[0];
  struct gk_app_ctx *app = ctx;
  double r = r_x(x, app->a_mid, app->x_inner);

  fout[0] = -app->r0/app->q0*alpha(r, -app->Lz/2.0, 0.0, ctx);
}

void bc_shift_func_up(double t, const double *xc, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xc[0];
  struct gk_app_ctx *app = ctx;
  double r = r_x(x, app->a_mid, app->x_inner);

  fout[0] = -app->r0/app->q0*alpha(r, app->Lz/2.0, 0.0, ctx);
}

void
calc_integrated_diagnostics(struct gkyl_tm_trigger* iot, gkyl_gyrokinetic_app* app, double t_curr, bool force_calc)
{
  if (gkyl_tm_trigger_check_and_bump(iot, t_curr) || force_calc) {
    gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
    gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);
  }
}

void
write_data(struct gkyl_tm_trigger* iot_conf, struct gkyl_tm_trigger* iot_phase,
  gkyl_gyrokinetic_app* app, double t_curr, bool force_write)
{
  bool trig_now_conf = gkyl_tm_trigger_check_and_bump(iot_conf, t_curr);
  if (trig_now_conf || force_write) {
    int frame = (!trig_now_conf) && force_write? iot_conf->curr : iot_conf->curr-1;

    gkyl_gyrokinetic_app_write_conf(app, t_curr, frame);

    gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
    gkyl_gyrokinetic_app_write_field_energy(app);

    gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);
    gkyl_gyrokinetic_app_write_integrated_mom(app);
  }

  bool trig_now_phase = gkyl_tm_trigger_check_and_bump(iot_phase, t_curr);
  if (trig_now_phase || force_write) {
    int frame = (!trig_now_conf) && force_write? iot_conf->curr : iot_conf->curr-1;

    gkyl_gyrokinetic_app_write_phase(app, t_curr, frame);
  }
}