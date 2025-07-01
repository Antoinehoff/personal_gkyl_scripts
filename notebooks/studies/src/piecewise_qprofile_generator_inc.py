import numpy as np
import matplotlib.pyplot as plt

class Context:
    def __init__(self, a_shift, Z_axis, R_axis, B_axis, R_LCFSmid, x_inner, x_outer, Nx, Npieces, qfunc):
        self.a_shift = a_shift
        self.Z_axis = Z_axis
        self.R_axis = R_axis
        self.B_axis = B_axis
        self.R_LCFSmid = R_LCFSmid
        self.x_inner = x_inner
        self.x_outer = x_outer
        self.Rmid_min = R_LCFSmid - x_inner
        self.Rmid_max = R_LCFSmid + x_outer
        self.R0 = 0.5 * (self.Rmid_min + self.Rmid_max)
        self.a_mid = R_LCFSmid - R_axis
        self.a_mid = R_axis / a_shift - np.sqrt(R_axis * (R_axis - 2 * a_shift * R_LCFSmid + 2 * a_shift * R_axis)) / a_shift
        self.r0 = self.R0 - R_axis
        self.Lx = self.Rmid_max - self.Rmid_min
        self.x_min = 0.0
        self.x_max = self.Lx
        self.Ly = 0.2
        self.Lz = 2.0 * np.pi - 1e-10
        self.Nx = Nx
        self.Npieces = Npieces
        self.qfunc = qfunc

    def r_x(self, x):
        return x + self.a_mid - self.x_inner

def shear(r, qprof, ctx):
    q = qprof(r, ctx)
    dqdr = np.gradient(q, r)
    return (r/q) * dqdr

def shift(x, qprofile, ctx):
    x0 = 0.5 * (ctx.x_min + ctx.x_max)
    r0 = ctx.r_x(x0)
    q0 = qprofile(r0, ctx)
    r = ctx.r_x(x)
    return -ctx.r0 / q0 * qprofile(r, ctx) * ctx.Lz

def fit_qprofiles(r, R, qprofile_func, Npieces, ctx):
    qprofile = qprofile_func(r, ctx)
    qfit_lin = np.polyfit(R, qprofile, 1)
    qfit_quad = np.polyfit(R, qprofile, 2)
    qfit_cub = np.polyfit(R, qprofile, 3)
    def qfit_lin_func(r, ctx):
        y = r + ctx.R_axis
        return qfit_lin[0] * y + qfit_lin[1]
    def qfit_quad_func(r, ctx):
        y = r + ctx.R_axis
        return qfit_quad[0] * y**2 + qfit_quad[1] * y + qfit_quad[2]
    def qfit_cub_func(r, ctx):
        y = r + ctx.R_axis
        return qfit_cub[0] * y**3 + qfit_cub[1] * y**2 + qfit_cub[2] * y + qfit_cub[3]
    qfit_piecewise = np.zeros((Npieces, 2))
    for i in range(Npieces):
        x1 = i * (ctx.x_max - ctx.x_min) / Npieces
        x2 = (i + 1) * (ctx.x_max - ctx.x_min) / Npieces
        r1 = ctx.r_x(x1)
        r2 = ctx.r_x(x2)
        R1 = r1 + ctx.R_axis
        R2 = r2 + ctx.R_axis
        q1 = qprofile_func(r1, ctx)
        q2 = qprofile_func(r2, ctx)
        qfit_piecewise[i, 0] = (q2 - q1) / (R2 - R1)
        qfit_piecewise[i, 1] = q1 - qfit_piecewise[i, 0] * R1
    def qfit_piecewise_func(r, ctx):
        if isinstance(r, float): 
            r = np.array([r])
        y = r + ctx.R_axis
        fit = np.zeros_like(y)
        for i in range(len(y)):
            for j in range(Npieces):
                x1 = j * (ctx.x_max - ctx.x_min) / Npieces
                x2 = (j + 1) * (ctx.x_max - ctx.x_min) / Npieces
                R1 = ctx.r_x(x1) + ctx.R_axis
                R2 = ctx.r_x(x2) + ctx.R_axis
                if y[i] >= R1 and y[i] <= R2:
                    fit[i] = qfit_piecewise[j, 0] * y[i] + qfit_piecewise[j, 1]
        return fit
    return qprofile, qfit_lin_func, qfit_quad_func, qfit_cub_func, qfit_piecewise_func, qfit_lin, qfit_quad, qfit_cub, qfit_piecewise

def print_piecewise_c_code(qfit_piecewise, Npieces, ctx):
    print("double qprofile(double r, double R_axis) {")
    print(" double R = r + R_axis;")
    for i in range(Npieces):
        x1 = i * (ctx.x_max - ctx.x_min) / Npieces
        x2 = (i + 1) * (ctx.x_max - ctx.x_min) / Npieces
        r1 = ctx.r_x(x1)
        r2 = ctx.r_x(x2)
        R1 = r1 + ctx.R_axis
        R2 = r2 + ctx.R_axis
        if i == 0:
            print(" if (R < %.14g) return %.14g * R + %.14g;" % (R2, qfit_piecewise[i, 0], qfit_piecewise[i, 1]))
        elif i == Npieces - 1:
            print(" if (R >= %.14g) return %.14g * R + %.14g;" % (R1, qfit_piecewise[i, 0], qfit_piecewise[i, 1]))
        else:
            print(" if (R >= %.14g && R < %.14g) return %.14g * R + %.14g;" % (R1, R2, qfit_piecewise[i, 0], qfit_piecewise[i, 1]))
    print("}")

def plot_qprofile_comparison(R, r, x, qprofile, qprofile_func, qfit_lin_func, qfit_quad_func, qfit_cub_func, qfit_piecewise_func, ctx):
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].plot(R, qprofile, '-k', label='Original TCV')
    ax[0].plot(R, qfit_lin_func(r, ctx), '-', label='Linear Fit')
    ax[0].plot(R, qfit_piecewise_func(r, ctx), '--c', label='Piecewise Linear Fit')
    ax[0].set_xlabel('Major Radius [m]')
    ax[0].set_ylabel('q-profile')
    ax[0].set_title('q-profile')
    shear_original = shear(r, qprofile_func, ctx)
    shear_linear = shear(r, qfit_lin_func, ctx)
    ax[1].plot(R, shear_original, '-k')
    ax[1].plot(R, shear_linear, '-')
    ax[1].plot(R, shear(r, qfit_piecewise_func, ctx), '--c')
    ax[1].set_xlabel('Major Radius [m]')
    ax[1].set_ylabel(r'$r/q(r) \times \partial q(r)/\partial r$')
    ax[1].set_title('Shear')
    shift_original = shift(x, qprofile_func, ctx)/ctx.Ly
    shift_linear = shift(x, qfit_lin_func, ctx)/ctx.Ly
    ax[2].plot(R, shift_original, '-k')
    ax[2].plot(R, shift_linear, '-')
    ax[2].plot(R, shift(x, qfit_piecewise_func, ctx)/ctx.Ly, '--c')
    ax[2].set_xlabel('Major Radius [m]')
    ax[2].set_ylabel(r'$S_y/L_y$')
    ax[2].set_title('Shift')
    ax[0].legend()
    plt.tight_layout()
    plt.show()

def run_qprofile_workflow(ctx, qprofile_func, print_code=False, plot=False, return_data=False):
    x = np.linspace(ctx.x_min, ctx.x_max, ctx.Nx)
    r = ctx.r_x(x)
    R = ctx.R_axis + r
    qprofile, qfit_lin_func, qfit_quad_func, qfit_cub_func, qfit_piecewise_func, qfit_lin, qfit_quad, qfit_cub, qfit_piecewise = fit_qprofiles(r, R, qprofile_func, ctx.Npieces, ctx)
    if print_code:
        print_piecewise_c_code(qfit_piecewise, ctx.Npieces, ctx)
    if plot:
        plot_qprofile_comparison(R, r, x, qprofile, qprofile_func, qfit_lin_func, qfit_quad_func, qfit_cub_func, qfit_piecewise_func, ctx)
    if return_data:
        return {
            'x': x,
            'r': r,
            'R': R,
            'qprofile': qprofile,
            'qfit_lin_func': qfit_lin_func,
            'qfit_quad_func': qfit_quad_func,
            'qfit_cub_func': qfit_cub_func,
            'qfit_piecewise_func': qfit_piecewise_func,
            'ctx': ctx
        }

## Compare PT and NT
def compare_PT_NT(profile_data_PT, profile_data_NT):
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    R_PT = profile_data_PT['R']
    R_NT = profile_data_NT['R']
    
    ax[0].plot(R_PT, profile_data_PT['qprofile'], '-r', label='PT Original')
    ax[0].plot(R_NT, profile_data_NT['qprofile'], '-b', label='NT Original')
    ax[0].set_xlabel('Major Radius [m]')
    ax[0].set_ylabel('q-profile')
    ax[0].set_title('q-profile Comparison')
    ax[0].legend()

    shear_PT = shear(profile_data_PT['r'], profile_data_PT['ctx'].qfunc, profile_data_PT['ctx'])
    shear_NT = shear(profile_data_NT['r'], profile_data_NT['ctx'].qfunc, profile_data_NT['ctx'])
    
    ax[1].plot(R_PT, shear_PT, '-r', label='PT Shear')
    ax[1].plot(R_NT, shear_NT, '-b', label='NT Shear')
    ax[1].set_xlabel('Major Radius [m]')
    ax[1].set_ylabel(r'$r/q(r) \times \partial q(r)/\partial r$')
    ax[1].set_title('Shear Comparison')
    ax[1].legend()

    shift_PT = shift(profile_data_PT['x'], profile_data_PT['ctx'].qfunc, profile_data_PT['ctx']) / profile_data_PT['ctx'].Ly
    shift_NT = shift(profile_data_NT['x'], profile_data_NT['ctx'].qfunc, profile_data_NT['ctx']) / profile_data_NT['ctx'].Ly
    
    ax[2].plot(R_PT, shift_PT, '-r', label='PT Shift')
    ax[2].plot(R_NT, shift_NT, '-b', label='NT Shift')
    ax[2].set_xlabel('Major Radius [m]')
    ax[2].set_ylabel(r'$S_y/L_y$')
    ax[2].set_title('Shift Comparison')
    ax[2].legend()

    plt.tight_layout()
    plt.show()