// Parameter default values
const defaultParameters = {
    // Q-profile parameters
    profile_type: "NT",
    Npieces: 8,
    poly_coeffs: "484.0615913225881, -1378.25993228584, 1309.3099150729233, -414.13270311478726",
    
    // TCV Discharge Parameters
    P_exp: 349200,
    a_shift: 1.0,
    kappa: 1.4,
    delta: -0.38,
    R_axis: 0.8867856264,
    R_LCFSmid: 1.0870056099999,
    Z_axis: 0.1414361745,
    B_axis: 1.4,
    
    // Reference Values
    Te0: 100,
    Ti0: 100,
    n0: "2.0e19",
    Bref: 1.129,
    
    // Numerical Resolutions
    num_cell_x: 48,
    num_cell_y: 32,
    num_cell_z: 16,
    num_cell_vpar: 12,
    num_cell_mu: 8,
    x_inner: 0.04,
    x_outer: 0.08,
    nrhos_y: 150,
    nvth_vpar_e: 6.0,
    nvth_vpar_i: 6.0,
    nvth_mu_e: 1.5,
    nvth_mu_i: 1.5,
    nuFrac: 1.0,
    
    // Time Parameters
    final_time: "2.e-3",
    num_frames: 1000
};

// Predefined polynomial coefficients for different configurations
const qProfileConfigs = {
    NT: {
        coeffs: [484.0615913225881, -1378.25993228584, 1309.3099150729233, -414.13270311478726],
        a_shift: 1.0,
        Z_axis: 0.1414361745,
        R_axis: 0.8867856264,
        B_axis: 1.4,
        R_LCFSmid: 1.0870056099999,
        delta: -0.38
    },
    PT: {
        coeffs: [497.3420166252413, -1408.736172826569, 1331.4134861681464, -419.00692601227627],
        a_shift: 0.25,
        Z_axis: 0.1414361745,
        R_axis: 0.8727315068,
        B_axis: 1.4,
        R_LCFSmid: 1.0968432365089495,
        delta: 0.35
    }
};

// C code template
const codeTemplate = `#include <math.h>
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
    // Collision parameters
    double nuFrac, nuElc, nuIon;
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
static void nuElc(double t, const double *xn, double *fout, void *ctx);
static void nuIon(double t, const double *xn, double *fout, void *ctx);
static void mapc2p(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx);
static void mapc2p_vel_elc(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx);
static void mapc2p_vel_ion(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx);
static void bmag_func(double t, const double *xn, double *bmag, void *ctx);
static void bc_shift_func_lo(double t, const double *xn, double *fout, void *ctx);
static void bc_shift_func_up(double t, const double *xn, double *fout, void *ctx);
static void write_data(struct gkyl_tm_trigger *iot_conf, struct gkyl_tm_trigger *iot_phase,
                       gkyl_gyrokinetic_app *app, double t_curr, bool force_write);
static void calc_integrated_diagnostics(struct gkyl_tm_trigger *iot, gkyl_gyrokinetic_app *app, double t_curr, bool force_write);

{{QPROFILE_CODE}}

struct gk_app_ctx create_ctx(void)
{
  // TCV #65130 ({{profile_type}}) discharge parameters.
  double P_exp     = {{P_exp}};        // P_sol measured [W]
  double a_shift   = {{a_shift}};             // Parameter in Shafranov shift.
  double kappa     = {{kappa}};             // Elongation (=1 for no elongation).
  double delta     = {{delta}};           // Triangularity (=0 for no triangularity).
  double R_axis    = {{R_axis}};    // Magnetic axis major radius [m].
  double R_LCFSmid = {{R_LCFSmid}}; // Major radius of the LCFS at the outboard midplane [m].
  double Z_axis    = {{Z_axis}};    // Magnetic axis height [m].
  double B_axis    = {{B_axis}};             // Magnetic field at the magnetic axis [T].
  // Note: after this line, all the parameters are simulation design parameters.

  // Reference values.
  double Te0  = {{Te0}}; // Ref. electron temperature [eV].
  double Ti0  = {{Ti0}}; // Ref. ion temperature [eV].
  double n0   = {{n0}};   // Ref. density [1/m^3]
  double Bref = {{Bref}};   // Ref. magnetic field [T].

  // Numerical resolutions.
  int num_cell_x = {{num_cell_x}};
  int num_cell_y = {{num_cell_y}};
  int num_cell_z = {{num_cell_z}};
  int num_cell_vpar = {{num_cell_vpar}};
  int num_cell_mu = {{num_cell_mu}};
  double x_inner = {{x_inner}}; // Width of the closed flux surface region [m].
  double x_outer = {{x_outer}}; // Width of the SOL region [m].
  double nrhos_y = {{nrhos_y}}; // Number of rho_s0 in the y-direction, used to set the domain size along y.
  double nvth_vpar_e = {{nvth_vpar_e}}; // sets vpar max e in terms of ref. elc th. vel.
  double nvth_vpar_i = {{nvth_vpar_i}}; // sets vpar max i in terms of ref. ion th. vel.
  double nvth_mu_e = {{nvth_mu_e}}; // sets mu max e in terms of mi*pow(4*vte0,2)/(2*B0).
  double nvth_mu_i = {{nvth_mu_i}}; // sets mu max i in terms of mi*pow(4*vti0,2)/(2*B0).
  double nuFrac = {{nuFrac}}; // Fraction of the Coulomb collision frequency to use in the simulation.

  // Time parameters
  double final_time = {{final_time}};
  int num_frames = {{num_frames}};
  double write_phase_freq = 0.2;
  int int_diag_calc_num = num_frames*100;
  double dt_failure_tol = 1.0e-3; // Minimum allowable fraction of initial time-step.
  int num_failures_max = 20; // Maximum allowable number of consecutive small time-steps.

{{TEMPLATE_ENDING}}
`;

// DOM elements
let tooltip;
let inputs = {};

function loadTemplateEnding() {
    return fetch('default_input.c')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to load template: ${response.status}`);
            }
            return response.text();
        })
        .then(text => {
            const marker = 'END OF THE USER PARAMETERS';
            const startIndex = text.indexOf(marker);
            
            if (startIndex !== -1) {
                return text.slice(startIndex);
            }
            
            // Return fallback if marker not found
            throw new Error('Template marker not found');
        })
        .catch(() => {
            // Fallback template ending
            return `  // END OF THE USER PARAMETERS
  //-------------------------------------------------------------

  // ... rest of the C code continues here ...
  
  return ctx;
}

// ... rest of the C functions ...`;
        });
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the app first so it's visible
    initializeApp();
});

function initializeApp() {
    // Get DOM elements
    tooltip = document.getElementById('tooltip');
    
    // Get all input elements
    Object.keys(defaultParameters).forEach(key => {
        inputs[key] = document.getElementById(key);
    });
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize with default profile
    handleProfileTypeChange();
    
    // Generate initial preview
    generateCodePreview();
}

function setupEventListeners() {
    // Tooltip events for labels
    const labels = document.querySelectorAll('label[data-tooltip]');
    labels.forEach(label => {
        label.addEventListener('mouseenter', showTooltip);
        label.addEventListener('mouseleave', hideTooltip);
        label.addEventListener('mousemove', moveTooltip);
    });
    
    // Profile type change event
    const profileTypeSelect = document.getElementById('profile_type');
    if (profileTypeSelect) {
        profileTypeSelect.addEventListener('change', handleProfileTypeChange);
    }
    
    // Input change events
    Object.values(inputs).forEach(input => {
        if (input) {
            input.addEventListener('input', generateCodePreview);
        }
    });
    
    // Button events
    document.getElementById('generate-qprofile').addEventListener('click', generateQProfile);
    document.getElementById('generate-file').addEventListener('click', generateCodePreview);
    document.getElementById('download-file').addEventListener('click', downloadFile);
    document.getElementById('reset-defaults').addEventListener('click', resetToDefaults);
}

function showTooltip(event) {
    const tooltipText = event.target.getAttribute('data-tooltip');
    tooltip.textContent = tooltipText;
    tooltip.classList.add('show');
    moveTooltip(event);
}

function hideTooltip() {
    tooltip.classList.remove('show');
}

function moveTooltip(event) {
    const tooltipRect = tooltip.getBoundingClientRect();
    const x = event.clientX;
    const y = event.clientY - tooltipRect.height - 10;
    
    tooltip.style.left = Math.max(10, Math.min(x - tooltipRect.width / 2, window.innerWidth - tooltipRect.width - 10)) + 'px';
    tooltip.style.top = Math.max(10, y) + 'px';
}

function getCurrentParameters() {
    const params = {};
    Object.keys(defaultParameters).forEach(key => {
        const input = inputs[key];
        if (input) {
            params[key] = input.value;
        } else {
            params[key] = defaultParameters[key];
        }
    });
    return params;
}

// Handle profile type changes
function handleProfileTypeChange() {
    const profileType = document.getElementById('profile_type').value;
    const customCoeffsDiv = document.getElementById('custom-coeffs');
    
    if (profileType === 'custom') {
        customCoeffsDiv.style.display = 'block';
    } else {
        customCoeffsDiv.style.display = 'none';
        
        // Update parameters based on selected configuration
        const config = qProfileConfigs[profileType];
        if (config) {
            document.getElementById('poly_coeffs').value = config.coeffs.join(', ');
            document.getElementById('a_shift').value = config.a_shift;
            document.getElementById('Z_axis').value = config.Z_axis;
            document.getElementById('R_axis').value = config.R_axis;
            document.getElementById('B_axis').value = config.B_axis;
            document.getElementById('R_LCFSmid').value = config.R_LCFSmid;
            document.getElementById('delta').value = config.delta;
        }
    }
    generateQProfile();
}

// Q-profile generation based on your piecewise generator
function generateQProfile() {
    const params = getCurrentParameters();
    const profileType = params.profile_type;
    
    let coeffs;
    if (profileType === 'custom') {
        coeffs = params.poly_coeffs.split(',').map(c => parseFloat(c.trim()));
    } else {
        coeffs = qProfileConfigs[profileType].coeffs;
    }
    
    if (coeffs.length !== 4) {
        showMessage('Error: Polynomial coefficients must be [a, b, c, d]', 'error');
        return;
    }
    
    const [a, b, c, d] = coeffs;
    const Npieces = parseInt(params.Npieces);
    const Nx = parseInt(params.num_cell_x);
    
    // Context creation (mimicking your Python Context class)
    const context = {
        a_shift: parseFloat(params.a_shift),
        Z_axis: parseFloat(params.Z_axis),
        R_axis: parseFloat(params.R_axis),
        B_axis: parseFloat(params.B_axis),
        R_LCFSmid: parseFloat(params.R_LCFSmid),
        x_inner: parseFloat(params.x_inner),
        x_outer: parseFloat(params.x_outer),
        delta: parseFloat(params.delta),
        Nx: Nx,
        Npieces: Npieces
    };
    
    context.Rmid_min = context.R_LCFSmid - context.x_inner;
    context.Rmid_max = context.R_LCFSmid + context.x_outer;
    context.R0 = 0.5 * (context.Rmid_min + context.Rmid_max);
    context.a_mid = context.R_LCFSmid - context.R_axis;
    context.a_mid = context.R_axis / context.a_shift - Math.sqrt(context.R_axis * (context.R_axis - 2 * context.a_shift * context.R_LCFSmid + 2 * context.a_shift * context.R_axis)) / context.a_shift;
    context.r0 = context.R0 - context.R_axis;
    context.Lx = context.Rmid_max - context.Rmid_min;
    context.x_min = 0.0;
    context.x_max = context.Lx;
    
    // Generate piecewise fit
    const qfit_piecewise = generatePiecewiseFit(a, b, c, d, context);
    
    // Generate C code
    const qprofileCode = generateQProfileCCode(qfit_piecewise, context, a, b, c, d);
    
    // Store the generated q-profile code globally for use in full file generation
    window.generatedQProfile = qprofileCode;
    
    showMessage('Q-profile generated successfully!', 'success');
    generateCodePreview(); // Update the full code preview
}

function generatePiecewiseFit(a, b, c, d, ctx) {
    const qfit_piecewise = [];
    
    function r_x(x) {
        return x + ctx.a_mid - ctx.x_inner;
    }
    
    function R_r(r) {
        return ctx.R_axis + r; // Simple r + R_axis mapping
    }
    
    function R_x(x) {
        return R_r(r_x(x));
    }
    
    function qfunc_original(R) {
        return a * Math.pow(R, 3) + b * Math.pow(R, 2) + c * R + d;
    }
    
    for (let i = 0; i < ctx.Npieces; i++) {
        const x1 = i * (ctx.x_max - ctx.x_min) / ctx.Npieces;
        const x2 = (i + 1) * (ctx.x_max - ctx.x_min) / ctx.Npieces;
        const R1 = R_x(x1);
        const R2 = R_x(x2);
        const q1 = qfunc_original(R1);
        const q2 = qfunc_original(R2);
        
        const slope = (q2 - q1) / (R2 - R1);
        const intercept = q1 - slope * R1;
        
        qfit_piecewise.push({
            slope: slope,
            intercept: intercept,
            R1: R1,
            R2: R2
        });
    }
    
    return qfit_piecewise;
}

function generateQProfileCCode(qfit_piecewise, ctx, a, b, c, d) {
    const profileType = document.getElementById('profile_type').value;
    const profileName = profileType === 'NT' ? 'TCV #65130 (NT)' : 
                       profileType === 'PT' ? 'TCV #65130 (PT)' : 'Custom';
    
    let code = '// Piecewise linear fit for q-profile ' + profileName + '\n';
    code += '// Context: a_shift=' + ctx.a_shift + ', Z_axis=' + ctx.Z_axis + ', R_axis=' + ctx.R_axis + ', B_axis=' + ctx.B_axis + ',\n';
    code += '//          R_LCFSmid=' + ctx.R_LCFSmid + ', x_inner=' + ctx.x_inner + ', x_outer=' + ctx.x_outer + ', Nx=' + ctx.Nx + '\n';
    code += '//          Npieces=' + ctx.Npieces + ', delta=' + ctx.delta + '\n';
    code += '// Polynomial coefficients: [' + a + ', ' + b + ', ' + c + ', ' + d + ']\n';
    code += 'static double qprofile(double R) {\n';
    
    for (let i = 0; i < qfit_piecewise.length; i++) {
        const piece = qfit_piecewise[i];
        if (i === 0) {
            code += ' if (R < ' + piece.R2.toExponential(12) + ') return ' + piece.slope.toExponential(12) + ' * R + ' + piece.intercept.toExponential(12) + ';\n';
        } else if (i === qfit_piecewise.length - 1) {
            code += ' if (R >= ' + piece.R1.toExponential(12) + ') return ' + piece.slope.toExponential(12) + ' * R + ' + piece.intercept.toExponential(12) + ';\n';
        } else {
            code += ' if (R >= ' + piece.R1.toExponential(12) + ' && R < ' + piece.R2.toExponential(12) + ') return ' + piece.slope.toExponential(12) + ' * R + ' + piece.intercept.toExponential(12) + ';\n';
        }
    }
    
    code += '}\n';
    return code;
}

async function generateCodePreview() {
    const params = getCurrentParameters();
    
    // Get the generated q-profile code or use default
    const qprofileCode = window.generatedQProfile || '// Piecewise linear fit for q-profile TCV #65130 (NT)\n// Context: a_shift=1.0, Z_axis=0.1414361745, R_axis=0.8867856264, B_axis=1.4,\n//          R_LCFSmid=1.0870056099999, x_inner=0.04, x_outer=0.08, Nx=48\n//          Npieces=8, delta=-0.38\nstatic double qprofile(double R) {\n if (R < 1.0918488304321) return 27.804896404549 * R + -27.92421756603;\n if (R >= 1.0918488304321 && R < 1.1068488304321) return 34.024085846787 * R + -34.714632284773;\n if (R >= 1.1068488304321 && R < 1.1218488304321) return 40.896758437384 * R + -42.321641903619;\n if (R >= 1.1218488304321 && R < 1.1368488304321) return 48.422914176186 * R + -50.764850916843;\n if (R >= 1.1368488304321 && R < 1.1518488304321) return 56.60255306331 * R + -60.063863819026;\n if (R >= 1.1518488304321 && R < 1.1668488304321) return 65.435675098706 * R + -70.238285104561;\n if (R >= 1.1668488304321 && R < 1.1818488304321) return 74.922280282424 * R + -81.307719267953;\n if (R >= 1.1818488304321) return 85.062368614383 * R + -93.291770803558;\n}';
    
    try {
        const codeEnding = await loadTemplateEnding();
        
        // Replace parameters in the template
        let code = codeTemplate.replace(/\{\{(\w+)\}\}/g, (match, paramName) => {
            if (paramName === 'TEMPLATE_ENDING') {
                return codeEnding;
            }
            return params[paramName] || defaultParameters[paramName] || match;
        });
        
        // Replace the q-profile placeholder
        code = code.replace('{{QPROFILE_CODE}}', qprofileCode);
        
        // Update the code preview
        const codePreview = document.getElementById('code-preview');
        if (codePreview) {
            codePreview.textContent = code;
        }
        
        // Update summary
        updateSummary(params);
        
        // Store the generated code globally for download
        window.generatedCode = code;
    } catch (error) {
        console.error('Error loading template ending:', error);
        // Use fallback template ending
        const fallbackEnding = `  // END OF THE USER PARAMETERS
  //-------------------------------------------------------------

  // ... rest of the C code continues here ...
  
  return ctx;
}

// ... rest of the C functions ...`;
        
        // Replace parameters in the template with fallback
        let code = codeTemplate.replace(/\{\{(\w+)\}\}/g, (match, paramName) => {
            if (paramName === 'TEMPLATE_ENDING') {
                return fallbackEnding;
            }
            return params[paramName] || defaultParameters[paramName] || match;
        });
        
        // Replace the q-profile placeholder
        code = code.replace('{{QPROFILE_CODE}}', qprofileCode);
        
        // Update the code preview
        const codePreview = document.getElementById('code-preview');
        if (codePreview) {
            codePreview.textContent = code;
        }
        
        // Update summary
        updateSummary(params);
        
        // Store the generated code globally for download
        window.generatedCode = code;
        
        showMessage('Warning: Using fallback template ending', 'info');
    }
}

function updateSummary(params) {
    // Update summary displays
    const summaryGrid = document.getElementById('summary-grid');
    const summaryTime = document.getElementById('summary-time');
    const summaryFrames = document.getElementById('summary-frames');
    const summaryPower = document.getElementById('summary-power');
    
    if (summaryGrid) {
        summaryGrid.textContent = params.num_cell_x + 'x' + params.num_cell_y + 'x' + params.num_cell_z;
    }
    
    if (summaryTime) {
        summaryTime.textContent = params.final_time + ' s';
    }
    
    if (summaryFrames) {
        summaryFrames.textContent = params.num_frames;
    }
    
    if (summaryPower) {
        summaryPower.textContent = (parseFloat(params.P_exp) / 1000).toFixed(1) + ' kW';
    }
}

function downloadFile() {
    const codeContent = window.generatedCode || '';
    if (!codeContent) {
        showMessage('No code to download. Please generate code first.', 'error');
        return;
    }
    
    const blob = new Blob([codeContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'input.c';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showMessage('File downloaded successfully!', 'success');
}

function resetToDefaults() {
    Object.keys(defaultParameters).forEach(key => {
        const input = inputs[key];
        if (input) {
            input.value = defaultParameters[key];
        }
    });
    
    // Reset q-profile
    window.generatedQProfile = null;
    
    // Reset to default profile type
    handleProfileTypeChange();
    
    generateCodePreview();
    showMessage('Parameters reset to defaults!', 'info');
}

// Show message function
function showMessage(message, type) {
    type = type || 'info';
    
    // Create a temporary message element
    const messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.style.position = 'fixed';
    messageDiv.style.top = '20px';
    messageDiv.style.right = '20px';
    messageDiv.style.padding = '15px 20px';
    messageDiv.style.borderRadius = '8px';
    messageDiv.style.color = 'white';
    messageDiv.style.fontWeight = '600';
    messageDiv.style.zIndex = '10000';
    messageDiv.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.2)';
    
    // Set background color based on type
    switch(type) {
        case 'success':
            messageDiv.style.background = 'linear-gradient(135deg, #27ae60, #229954)';
            break;
        case 'error':
            messageDiv.style.background = 'linear-gradient(135deg, #e74c3c, #c0392b)';
            break;
        case 'info':
        default:
            messageDiv.style.background = 'linear-gradient(135deg, #3498db, #2980b9)';
            break;
    }
    
    document.body.appendChild(messageDiv);
    
    // Remove after 3 seconds
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.parentNode.removeChild(messageDiv);
        }
    }, 3000);
}

// Validate inputs
function validateInputs() {
    let isValid = true;
    const errors = [];
    
    // Check for required numeric fields
    Object.keys(defaultParameters).forEach(key => {
        const input = inputs[key];
        if (input && input.type === 'number') {
            const value = parseFloat(input.value);
            if (isNaN(value)) {
                isValid = false;
                errors.push(key + ' must be a valid number');
            }
        }
    });
    
    // Check for positive values where needed
    const positiveFields = ['num_cell_x', 'num_cell_y', 'num_cell_z', 'num_cell_vpar', 'num_cell_mu', 'num_frames'];
    positiveFields.forEach(key => {
        const input = inputs[key];
        if (input && parseFloat(input.value) <= 0) {
            isValid = false;
            errors.push(key + ' must be positive');
        }
    });
    
    if (!isValid) {
        showMessage('Validation errors: ' + errors.join(', '), 'error');
    }
    
    return isValid;
}
