import numpy as np
import matplotlib.pyplot as plt

figsize = (9, 3)

class Context:
    def __init__(self, a_shift, Z_axis, R_axis, B_axis, R_LCFSmid, delta, x_inner, x_outer, Nx, Npieces, qfunc, Rfunc='r+Raxis'):
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
        self.r0 = self.R0 - R_axis
        self.delta = delta
        self.Lx = self.Rmid_max - self.Rmid_min
        self.x_min = 0.0
        self.x_max = self.Lx
        self.Ly = 0.2
        self.Lz = 2.0 * np.pi - 1e-10
        self.Nx = Nx
        self.Npieces = Npieces
        self.qfunc = qfunc
        self.Rfunc = Rfunc
    
    def a_mid(self):
        return self.R_axis / self.a_shift * (1 - np.sqrt(1 + 2*self.a_shift*(self.R_axis - self.R_LCFSmid)/self.R_axis))
    
    def r_x(self, x):
        return (x - self.x_inner) + self.a_mid()
    
    def Rrt0(self, r):
        a_shift = self.a_shift
        R_axis = self.R_axis
        return R_axis - a_shift*r*r/(2.*R_axis)
    
    def R_rtheta(self, r, theta):
        delta = self.delta
        return self.Rrt0(r) + r*np.cos(theta + np.arcsin(delta)*np.sin(theta))
    
    def R_rRaxis(self,r):
        return self.R_axis + r
    
    def R_r(self,r):
        if self.Rfunc == 'r+Raxis':
            return self.R_rRaxis(r)
        else :
            return self.R_rtheta(r, 0.0)
    
    def R_x(self, x):
        return self.R_r(self.r_x(x))
        # return self.Rrt0(self.a_mid()) + self.r_x(x)

def shear(x, qfunc, ctx):
    qeval = qfunc(ctx.R_x(x), ctx)
    r = ctx.r_x(x)
    dqdr = np.gradient(qeval, r)
    return (r/qeval) * dqdr

def shift(x, qfunc, ctx):
    x0 = 0.5 * (ctx.x_min + ctx.x_max)
    r0 = ctx.r_x(x0)
    R0 = ctx.R_r(r0)
    q0 = qfunc(R0, ctx)
    R = ctx.R_x(x)
    return -ctx.r0 / q0 * qfunc(R, ctx) * ctx.Lz

def fit_qprofiles(R, qfunc_original, Npieces, ctx):
    qeval_original = qfunc_original(R, ctx)
    
    qfit_piecewise = np.zeros((Npieces, 2))
    for i in range(Npieces):
        x1 = i * (ctx.x_max - ctx.x_min) / Npieces
        x2 = (i + 1) * (ctx.x_max - ctx.x_min) / Npieces
        R1 = ctx.R_x(x1)
        R2 = ctx.R_x(x2)
        q1 = qfunc_original(R1, ctx)
        q2 = qfunc_original(R2, ctx)
        qfit_piecewise[i, 0] = (q2 - q1) / (R2 - R1)
        qfit_piecewise[i, 1] = q1 - qfit_piecewise[i, 0] * R1
        
    def qfunc_piecewise(R, ctx):
        if isinstance(R, float): 
            R = np.array([R])
        fit = np.zeros_like(R)
        for i in range(len(R)):
            for j in range(Npieces):
                x1 = j * (ctx.x_max - ctx.x_min) / Npieces
                x2 = (j + 1) * (ctx.x_max - ctx.x_min) / Npieces
                R1 = ctx.R_x(x1)
                R2 = ctx.R_x(x2)
                if R[i] >= R1 and R[i] <= R2:
                    fit[i] = qfit_piecewise[j, 0] * R[i] + qfit_piecewise[j, 1]
                    break
            else:
                # Handle points outside the range
                if R[i] < ctx.R_x(0):
                    fit[i] = qfit_piecewise[0, 0] * R[i] + qfit_piecewise[0, 1]
                else:
                    fit[i] = qfit_piecewise[-1, 0] * R[i] + qfit_piecewise[-1, 1]
                    
        return fit
    return qeval_original, qfunc_piecewise, qfit_piecewise

def print_piecewise_c_code(qfit_piecewise, Npieces, ctx):
    print(f"// Piecewise linear fit for q-profile ({Npieces} pieces)")
    print(f"// Context: a_shift={ctx.a_shift}, Z_axis={ctx.Z_axis}, R_axis={ctx.R_axis}, B_axis={ctx.B_axis},")
    print(f"//          R_LCFSmid={ctx.R_LCFSmid}, x_inner={ctx.x_inner}, x_outer={ctx.x_outer}, Nx={ctx.Nx}")
    print(f"//          Npieces={ctx.Npieces}, delta={ctx.delta}")
    print("static double qprofile(double R) {")
    for i in range(Npieces):
        x1 = i * (ctx.x_max - ctx.x_min) / Npieces
        x2 = (i + 1) * (ctx.x_max - ctx.x_min) / Npieces
        R1 = ctx.R_x(x1)
        R2 = ctx.R_x(x2)
        if i == 0:
            print(" if (R < %.14g) return %.14g * R + %.14g;" % (R2, qfit_piecewise[i, 0], qfit_piecewise[i, 1]))
        elif i == Npieces - 1:
            print(" if (R >= %.14g) return %.14g * R + %.14g;" % (R1, qfit_piecewise[i, 0], qfit_piecewise[i, 1]))
        else:
            print(" if (R >= %.14g && R < %.14g) return %.14g * R + %.14g;" % (R1, R2, qfit_piecewise[i, 0], qfit_piecewise[i, 1]))
    print("}")

def plot_qprofile_comparison(R, x, qeval_original, qfunc_original, qfunc_piecewise, ctx):
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    # x1 = (R - ctx.R_axis) / ctx.a_mid()
    # RLCFS = (ctx.R_LCFSmid - ctx.R_axis) / ctx.a_mid()
    x1 = (R - ctx.R_axis) / (ctx.R_LCFSmid - ctx.R_axis)
    RLCFS = (ctx.R_LCFSmid - ctx.R_axis) / (ctx.R_LCFSmid - ctx.R_axis)
    xlabel = r'$\rho$'
    
    ax[0].plot(x1, qeval_original, '-k', label='Original TCV')
    ax[0].plot(x1, qfunc_piecewise(R, ctx), '--c', label='Piecewise Linear Fit')
    ax[0].axvline(RLCFS, color='gray', linestyle='--', label='LCFS')
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel('q-profile')
    ax[0].set_title('q-profile')
    
    ax[1].plot(x1, shear(x, qfunc_original, ctx), '-k')
    ax[1].plot(x1, shear(x, qfunc_piecewise, ctx), '--c')
    ax[1].axvline(RLCFS, color='gray', linestyle='--', label='LCFS')
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(r'$r/q(r) \times \partial q(r)/\partial r$')
    ax[1].set_title('Shear')
    
    ax[2].plot(x1, shift(x, qfunc_original, ctx)/ctx.Ly, '-k')
    ax[2].plot(x1, shift(x, qfunc_piecewise, ctx)/ctx.Ly, '--c')
    ax[2].axvline(RLCFS, color='gray', linestyle='--', label='LCFS')
    ax[2].set_xlabel(xlabel)
    ax[2].set_ylabel(r'$S_y/L_y$')
    ax[2].set_title('Shift')
    ax[0].legend()
    plt.tight_layout()
    plt.show()

def run_qprofile_workflow(ctx, qfunc_original, print_code=False, plot=False, return_data=False):
    x = np.linspace(ctx.x_min, ctx.x_max, ctx.Nx)
    r = ctx.r_x(x)
    R = ctx.R_x(x)
    Rsim = ctx.R_rtheta(r, 0.0)
    qeval_original, qfunc_piecewise, qfit_piecewise = fit_qprofiles(R, qfunc_original, ctx.Npieces, ctx)
    
    shear_original = shear(x, qfunc_original, ctx)
    shear_piecewise = shear(x, qfunc_piecewise, ctx)
    
    shift_original = shift(x, qfunc_original, ctx)/ctx.Ly
    shift_piecewise = shift(x, qfunc_piecewise, ctx)/ctx.Ly
    
    if print_code: print_piecewise_c_code(qfit_piecewise, ctx.Npieces, ctx)
    if plot: plot_qprofile_comparison(Rsim, x, qeval_original, qfunc_original, qfunc_piecewise, ctx)
    if return_data:
        return {
            'x': x,
            'r': r,
            'R': R,
            'Rsim': Rsim,
            'qeval_original': qeval_original,
            'qfunc_original': qfunc_original,
            'qfunc_piecewise': qfunc_piecewise,
            'shear_original': shear_original,
            'shear_piecewise': shear_piecewise,
            'shift_original': shift_original,
            'shift_piecewise': shift_piecewise,
            'ctx': ctx,
            'label': ctx.qfunc.__name__
        }

def compare_qprofs(profile_data_1, profile_data_2):
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    R_1 = profile_data_1['Rsim']
    R_2 = profile_data_2['Rsim']
    
    ax[0].plot(R_1, profile_data_1['qeval_original'], '-r', label=profile_data_1['label'])
    ax[0].axvline(profile_data_1['ctx'].R_LCFSmid, color='gray', linestyle='-', label=f'{profile_data_1['label']} LCFS')
    ax[0].plot(R_2, profile_data_2['qeval_original'], '--b', label=profile_data_2['label'])
    ax[0].axvline(profile_data_2['ctx'].R_LCFSmid, color='gray', linestyle='--', label=f'{profile_data_2['label']} LCFS')
    ax[0].set_xlabel('Major Radius [m]')
    ax[0].set_ylabel('q-profile')
    ax[0].set_title('q-profile Comparison')
    ax[0].legend()

    ax[1].plot(R_1, profile_data_1['shear_original'], '-r', label=profile_data_1['label'])
    ax[1].axvline(profile_data_1['ctx'].R_LCFSmid, color='gray', linestyle='-', label=f'{profile_data_1['label']} LCFS')
    ax[1].plot(R_2, profile_data_2['shear_original'], '--b', label=profile_data_2['label'])
    ax[1].axvline(profile_data_2['ctx'].R_LCFSmid, color='gray', linestyle='--', label= f'{profile_data_2['label']} LCFS')
    ax[1].set_xlabel('Major Radius [m]')
    ax[1].set_ylabel(r'$r/q(r) \times \partial q(r)/\partial r$')
    ax[1].set_title('Shear Comparison')

    ax[2].plot(R_1, profile_data_1['shift_original'], '-r', label=profile_data_1['label'])
    ax[2].axvline(profile_data_1['ctx'].R_LCFSmid, color='gray', linestyle='-', label=f'{profile_data_1['label']} LCFS')
    ax[2].plot(R_2, profile_data_2['shift_original'], '--b', label=profile_data_2['label'])
    ax[2].axvline(profile_data_2['ctx'].R_LCFSmid, color='gray', linestyle='--', label=f'{profile_data_2['label']} LCFS')
    ax[2].set_xlabel('Major Radius [m]')
    ax[2].set_ylabel(r'$S_y/L_y$')
    ax[2].set_title('Shift Comparison')

    plt.tight_layout()
    plt.show()