import numpy as np
x0 = 0.025
xmin = 0
xmax = 0.12
fraction = (x0 - xmin)/(xmax - xmin)
frame_slice = sim_frames

time = np.zeros(len(frame_slice))
phi_ktheta_t = np.zeros((len(frame_slice), polproj.nzI//2 + 1), dtype=complex)
ExB_vth_cut_avg_t = np.zeros(len(frame_slice))
ExB_vr_cut_avg_t = np.zeros(len(frame_slice))
for (it, frame) in enumerate(frame_slice):

    field_RZ, RR, ZZ, time_norm = polproj.get_projection(fieldName='phi',timeFrame=frame)
    Bmag_RZ, _, _, _ = polproj.get_projection(fieldName='Bmag',timeFrame=frame)
    # print(f"Dimensions of the projection grid: RR {RR.shape}, ZZ {ZZ.shape}, field_RZ {field_RZ.shape}")
    time[it] = time_norm * simulation.normalization.dict['tscale']
    NR = RR.shape[0]
    NZ = RR.shape[1]
    icut = int(fraction * NR)

    # compute the radial coordinate r from the inner flux surface
    lx = np.zeros_like(RR)
    ly = np.zeros_like(RR)
    for i in range(NR):
        lx[i,1:] = np.diff(RR[i,:])
        ly[i,1:] = np.diff(ZZ[i,:])
    dl = np.sqrt( lx**2 + ly**2 )
    ll = np.zeros_like(RR)
    for i in range(NR):
        for j in range(NZ-1):
            ll[i,j+1] = ll[i,j] + dl[i,j]
    # print(f"Length of the cut line at x={x0} : L = {ll[icut,-1]:.4f} m")

    # compute the radial coordinate r from the inner flux surface
    rx = np.zeros_like(RR)
    ry = np.zeros_like(RR)
    for j in range(NZ):
        rx[1:,j] = np.diff(RR[:,j])
        ry[1:,j] = np.diff(ZZ[:,j])
    dr = np.sqrt( rx**2 + ry**2 )
    rr = np.zeros_like(RR)
    for j in range(NZ):
        for i in range(NR-1):
            rr[i+1,j] = rr[i,j] + dr[i,j]
    # print(f"Maximum radial distance from the inner flux surface : r_max = {rr[-1,-1]:.4f} m")

    phi_y = field_RZ[icut,:]
    r = rr[icut,:]
    l = ll[icut,:]

    # Find the pitch angle of the field line factor
    r0 = simulation.geom_param.r_x(x0)
    R0 = simulation.geom_param.R_x(x0)
    q0 = simulation.geom_param.qprofile_x(x0)
    pitch_factor = 1/(1 + (r0/(R0*q0)))
    # print(f"Pitch factor at x={x0} : {pitch_factor:.4f}")
    # We assume that B = B_toroidal e_phi + B_poloidal e_theta
    # ~= |B| * ( pitch_factor * e_phi + (1-pitch_factor) * e_theta )
    # Now we compute the components of the ExB velocity
    # v_E = - grad(phi) x B / |B|^2 = - (1/|B|) * ( grad(phi) x ( pitch_factor * e_phi + (1-pitch_factor) * e_theta ) )
    # = - (1/|B|) * ( pitch_factor * grad(phi) x e_phi + (1-pitch_factor) * grad(phi) x e_theta )
    # = - (1/|B|) * ( pitch_factor * ( dphi/dr * e_r x e_phi + dphi/dtheta * e_theta x e_phi ) + (1-pitch_factor) * ( dphi/dr * e_r x e_theta + dphi/dtheta * e_theta x e_theta ) )
    # = - (1/|B|) * ( pitch_factor * ( dphi/dr * (-e_theta) + dphi/dtheta * e_r ) + (1-pitch_factor) * ( dphi/dr * e_phi + 0 ) )
    # = - (1/|B|) * ( pitch_factor * dphi/dtheta * e_r - pitch_factor * dphi/dr * e_theta + (1-pitch_factor) * dphi/dr * e_phi )
    # Radial ExB velocity component:
    # v_Er = - (1/|B|) * pitch_factor * dphi/dtheta
    # Poloidal ExB velocity component:
    # v_Etheta = (1/|B|) * pitch_factor * dphi/dr
    # ExB_vth = np.zeros_like(field_RZ)
    ExB_vth_cut = np.zeros(NZ)
    for j in range(NZ-1):
        Bmag = Bmag_RZ[icut,j]
        phim1 = field_RZ[icut-1,j]
        phip1 = field_RZ[icut+1,j]
        dr = rr[icut+1,j] - rr[icut-1,j]
        dphidr = (phip1 - phim1) / dr
        ExB_vth_cut[j] = pitch_factor/Bmag * dphidr

    # ExB_vth_cut_avg_t[it] = np.mean( ExB_vth[1:-1] )
    ExB_vth_cut_avg_t[it] = np.mean(ExB_vth_cut)
    print(f"Average poloidal ExB velocity at x={x0} : <v_Etheta> = {ExB_vth_cut_avg_t[it]:.4f} m/s")

    phi_theta = field_RZ[icut,:]
    rtheta = ll[icut,:]

    phi_ktheta_t[it,:] = np.fft.rfft( phi_theta )

from scipy.signal import windows
# Now force periodicity in time with a Kaiser window
beta = 8
N_t = phi_ktheta_t.shape[0]
kaiser_window = windows.kaiser(N_t, beta)
phi_ktheta_t_keiser = phi_ktheta_t * kaiser_window[:, np.newaxis]


Ntheta = len(rtheta)
dl = (rtheta[1] - rtheta[0])
L = (rtheta[-1] - rtheta[0])
ktheta_min = 2 * np.pi / L
ktheta_max = Ntheta * 2 * np.pi / L
ktheta =  np.linspace(ktheta_min, ktheta_max, Ntheta//2 + 1)

Ntime = len(time)
dt = (time[1] - time[0])
T = (time[-1] - time[0])
omega_min = 2 * np.pi / T
omega_max = Ntime * np.pi / T
omega =  np.linspace(-omega_max, omega_max, Ntime)

print(f"ktheta range: [{ktheta_min:.4f}, {ktheta_max:.4f}] 1/m")
print(f"omega range: [{-omega_max:.4f}, {omega_max:.4f}] rad/s")
print(f"dt = {dt:.4e} s, T = {T:.4e} s")
print(f"velocity range: [{omega_min/ktheta_max:.4f}, {omega_max/ktheta_min:.4f}] m/s")


# Introduce the factor to compensate the poloidal rotation
ExB_vth_cut_avg = np.mean(ExB_vth_cut_avg_t)
kk, tt = np.meshgrid( ktheta, time )
ExB_vth_cut_tt = ExB_vth_cut_avg_t[:, np.newaxis] * np.ones_like(kk)
ExB_rot_fact = np.exp( 1j * kk * -ExB_vth_cut_tt * tt )
phi_ktheta_t_keiser_shift = phi_ktheta_t_keiser * ExB_rot_fact

phi_ktheta_omega_keiser = np.fft.fftshift( np.fft.fft( phi_ktheta_t_keiser, axis=0 ), axes=0 )
phi_ktheta_omega_keiser_shift = np.fft.fftshift( np.fft.fft( phi_ktheta_t_keiser_shift, axis=0 ), axes=0 )


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

kk, ww = np.meshgrid( ktheta, omega )
#ww = ww - kk * ExB_vth_cut_avg
rho0 = r0/simulation.geom_param.a_mid
rho_s = simulation.get_rho_s()
c_s = simulation.get_c_s()
R = simulation.geom_param.R_axis
omega_d = c_s / R
kk *= rho_s
ww /= omega_d
# Plot the spectrogram with a logarithmic colormap
plt.figure(figsize=(8, 4))
plt.pcolormesh(kk, ww, np.abs(phi_ktheta_omega_keiser_shift)**2, shading='auto', 
               norm=LogNorm(vmin=1e4, vmax=5e7), cmap='inferno')
plt.colorbar(label=r'$|\phi_{k_y,\omega}|^2$')
plt.xlabel(r'$k_\theta \rho_s$')
plt.xlim(0.04,1)
plt.ylim(-8,8)

# Force ylabel to use power of ten at the top
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.ylabel(r'$\omega R/c_s$')
plt.title(r'Spectrogram of $\phi$ at $\rho=$' + f'{rho0:.2f}' + r', compensated for $\langle v_{ExB,\theta} \rangle_{t,\theta} = $' + f'{ExB_vth_cut_avg:.2f}' + ' m/s')
plt.show()
