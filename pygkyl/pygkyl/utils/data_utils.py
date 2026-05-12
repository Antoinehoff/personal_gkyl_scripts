import numpy as np
from scipy.interpolate import interp1d
from ..classes import Frame, TimeSerie

def get_2D_movie_time_serie(simulation,cut_dir,cut_coord,time_frames, fieldnames, fluctuation, fourier_y):
    # Load all data for the movie e.g. [[n0,u0,...],[n1,u1,...],...]
    movie_frames = [[None for _ in range(len(fieldnames))] for _ in range(len(time_frames))]
    fmin = np.zeros_like(fieldnames, dtype=float)
    fmax = np.zeros_like(fieldnames, dtype=float)
    absfmax = np.zeros_like(fieldnames, dtype=float)
    
    for kf,field in enumerate(fieldnames):
        serie = TimeSerie(simulation=simulation, fieldname=field, time_frames=time_frames, load=True, fourier_y=fourier_y)
        serie.slice(cut_dir, cut_coord)
        if len(fluctuation) > 0: serie.fluctuations(fluctuationType=fluctuation)
        # add the frames to the movie
        for it in range(len(time_frames)):
            movie_frames[it][kf] = serie.frames[it].copy()

    # compute max min and maxabs for each fields
    for it,tf in enumerate(time_frames):
        for kf,field in enumerate(fieldnames):
            fmin[kf] = min(fmin[kf], np.min(movie_frames[it][kf].values))
            fmax[kf] = max(fmax[kf], np.max(movie_frames[it][kf].values))
            absfmax[kf] = max(absfmax[kf], np.max(np.abs(movie_frames[it][kf].values)))
    vlims = [[fmin[kf],fmax[kf]] if fmin[kf] >= 0 else [-absfmax[kf],absfmax[kf]] for kf in range(len(fieldnames))]
    return movie_frames, vlims

def get_minmax_values(field = None, simulation=None, fieldname=None, time_frames=None):
    if field is not None:
        fmin = np.min(field)
        fmax = np.max(field)
        nx_mid = field.shape[0]//2
        nz_mid = field.shape[2]//2
        f_OBMP_SOL_min = np.min(field[nx_mid:,:,nz_mid])
        f_OBMP_SOL_max = np.max(field[nx_mid:,:,nz_mid])
        return [fmin, fmax], [f_OBMP_SOL_min, f_OBMP_SOL_max]
    else:
        for tf in time_frames:
            frame = Frame(simulation, fieldname, tf, load=True)
            fmin = np.min(frame.values) if tf == time_frames[0] else min(fmin, np.min(frame.values))
            fmax = np.max(frame.values) if tf == time_frames[0] else max(fmax, np.max(frame.values))
            nx_mid = frame.values.shape[0]//2
            nz_mid = frame.values.shape[2]//2
            f_OBMP_SOL = frame.values[nx_mid:,:,nz_mid]
            f_OBMP_SOL_min = np.min(f_OBMP_SOL) if tf == time_frames[0] else min(f_OBMP_SOL_min, np.min(f_OBMP_SOL))
            f_OBMP_SOL_max = np.max(f_OBMP_SOL) if tf == time_frames[0] else max(f_OBMP_SOL_max, np.max(f_OBMP_SOL))
    return [fmin, fmax], [f_OBMP_SOL_min, f_OBMP_SOL_max]

def get_1xt_diagram(simulation, fieldname, cutdirection, ccoords,
                    tfs):
    tfs = [tfs] if not isinstance(tfs,list) else tfs
    fourier_y = cutdirection.find('k') > -1
    cutdirection = cutdirection.replace('k','')
    # to store times and values
    # get number of time frames
    nt = len(tfs)
    # get number of values
    f0 = Frame(simulation,fieldname,tfs[0],load=True, fourier_y=fourier_y)
    f0.slice(cutdirection,ccoords)
    # get number of values
    nv = np.squeeze(f0.values.size)
    t  = np.zeros(nt)
    values = np.zeros((nv,nt))
    # Fill ZZ with data for each time frame
    for it,tf in enumerate(tfs):
        frame = Frame(simulation,fieldname,tf,load=True, fourier_y=fourier_y)
        frame.slice(cutdirection,ccoords)
        t[it] = frame.time
        values[:,it] = np.squeeze(frame.values)
    if values.ndim == 1: values = values.reshape(1,-1)
    x = frame.new_grids[0]
    tsymb = simulation.normalization.dict['tsymbol'] 
    tunit = simulation.normalization.dict['tunits']
    tlabel = tsymb+(' ['+tunit+']')*(1-(tunit==''))
    xlabel = frame.new_gsymbols[0]+(' ['+frame.new_gunits[0]+']')*(1-(frame.new_gunits[0]==''))
    vlabel = frame.vsymbol+(' ['+frame.vunits+']')*(1-(frame.vunits==''))
    slicetitle = frame.slicetitle
    return x,t,values,xlabel,tlabel,vlabel,frame.vunits,slicetitle, fourier_y


def _double_fft(phi_space_t, coord_phys, time_phys, norm, k_key):
    """
    Compute a 2D power spectrum via FFT along space (one-sided) then time.

    Parameters
    ----------
    phi_space_t : ndarray, shape (Nspace, Nt)
        Field values on a uniform physical grid.
    coord_phys : ndarray, shape (Nspace,)
        Physical spatial coordinate in metres (must be uniformly spaced).
    time_phys : ndarray, shape (Nt,)
        Physical time in seconds.
    norm : Normalization
        Simulation normalization object.
    k_key : str
        Key for the wavenumber in *norm.dict* (e.g. ``'ky'`` or ``'ktheta'``).

    Returns
    -------
    k_norm : ndarray, shape (Nk,)
        Normalised wavenumber axis (one-sided, k >= 0).
    omega_norm : ndarray, shape (Nt,)
        Normalised frequency axis (zero-centred via fftshift).
    power : ndarray, shape (Nk, Nt)
        Power spectral density ``|phi_k_omega|**2``.
    """
    Nspace, Nt = phi_space_t.shape

    dc_phys = np.mean(np.diff(coord_phys))  # physical spatial spacing [m]
    dt_phys = np.mean(np.diff(time_phys))   # physical time spacing [s]

    # Kaiser window on time axis to suppress spectral leakage
    win_t = np.kaiser(Nt, beta=8)
    phi_windowed = phi_space_t * win_t[np.newaxis, :]

    # Spatial FFT (one-sided, real input)
    phi_k_t = np.fft.rfft(phi_windowed, axis=0)

    # Time FFT + fftshift to place zero frequency at centre
    phi_k_omega = np.fft.fftshift(np.fft.fft(phi_k_t, axis=1), axes=1)

    # Wavenumber axis [rad/m], then normalize
    k_phys = np.fft.rfftfreq(Nspace, d=dc_phys) * 2.0 * np.pi  # rad/m
    k_scale = norm.dict[k_key + 'scale']   # e.g. 1/rho_i  →  k_norm = k*rho_i
    k_norm = k_phys / k_scale

    # Frequency axis [rad/s], then normalize to omega*R/cs
    omega_phys = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt_phys)) * 2.0 * np.pi  # rad/s
    omega_scale = norm.dict['omegascale']  # R/cs  [s]
    omega_norm = omega_phys * omega_scale  # dimensionless omega*R/cs

    power = np.abs(phi_k_omega) ** 2
    # Drop the k=0 (DC) mode and normalise so the total power sums to 1
    k_norm = k_norm[1:]
    power  = power[1:]
    total  = power.sum()
    if total > 0:
        power = power / total
    return k_norm, omega_norm, power


def get_toroidal_spectrum(simulation, fieldname, cut_coords, twindow):
    """
    Build the toroidal (ky, omega) power spectrum of a field.

    Slices the field along the binormal *y* direction at the given (x, z)
    cut coordinates, accumulates data over *twindow* frames, then computes
    a 2D FFT in (y, t) space.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    fieldname : str
        Name of the field to analyse (e.g. ``'phi'``).
    cut_coords : list of float
        Physical coordinates for the cut ``[x0, z0]`` in SI units.
    twindow : list of int
        List of time-frame indices to include.

    Returns
    -------
    k_norm : ndarray
        Normalised binormal wavenumber axis ``ky * rho_i`` (one-sided).
    omega_norm : ndarray
        Normalised frequency axis ``omega * R/cs`` (zero-centred).
    power : ndarray, shape (Nk, Nt)
        Power spectral density.
    k_label : str
        Axis label for the wavenumber.
    omega_label : str
        Axis label for the frequency.
    vlabel : str
        Colour-bar label (field symbol and units).
    slicetitle : str
        Slice description for the figure title.
    """
    twindow = [twindow] if not isinstance(twindow, list) else twindow
    norm = simulation.normalization

    # First frame: determine array dimensions and labels
    f0 = Frame(simulation, fieldname, twindow[0], load=True, fourier_y=False)
    f0.slice('y', cut_coords)
    Ny = np.squeeze(f0.values).size
    Nt = len(twindow)

    phi_y_t = np.zeros((Ny, Nt))
    t_norm  = np.zeros(Nt)

    for it, tf in enumerate(twindow):
        frame = Frame(simulation, fieldname, tf, load=True, fourier_y=False)
        frame.slice('y', cut_coords)
        phi_y_t[:, it] = np.squeeze(frame.values)
        t_norm[it]     = frame.time

    # Convert normalised grids to physical units for FFT
    yscale = norm.dict['yscale']   # rho_i [m]
    tscale = norm.dict['tscale']   # [s] per normalised time unit
    yshift = norm.dict['yshift']
    y_norm = f0.new_grids[0]
    y_phys = (y_norm + yshift) * yscale   # [m]
    t_phys = t_norm * tscale              # [s]

    k_norm, omega_norm, power = _double_fft(phi_y_t, y_phys, t_phys, norm, k_key='ky')

    ky_sym    = norm.dict['kysymbol']
    ky_unit   = norm.dict['kyunits']
    k_label   = ky_sym + (' [' + ky_unit + ']') * (ky_unit != '')
    omega_sym  = norm.dict['omegasymbol']
    omega_unit = norm.dict['omegaunits']
    omega_label = omega_sym + (' [' + omega_unit + ']') * (omega_unit != '')
    vlabel    = f0.vsymbol + (' [' + f0.vunits + ']') * (f0.vunits != '')
    slicetitle = f0.slicetitle

    return k_norm, omega_norm, power, k_label, omega_label, vlabel, slicetitle


def get_poloidal_spectrum(simulation, fieldname, cut_x, twindow, polproj,
                          exb_correction=False):
    """
    Build the poloidal (ktheta, omega) power spectrum of a field.

    Uses :class:`~pygkyl.projections.PoloidalProjection` to project the
    field onto the (R, Z) poloidal plane, extracts a 1D cut at the radial
    position *cut_x*, resamples onto a uniform arc-length grid (fixing the
    non-uniform spacing issue), then computes a 2D FFT in (theta, t) space.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    fieldname : str
        Name of the field to analyse (e.g. ``'phi'``).
    cut_x : float
        Physical radial coordinate for the poloidal cut [m].
    twindow : list of int
        List of time-frame indices to include.
    polproj : PoloidalProjection
        Already set-up poloidal projection object.
    exb_correction : bool, optional
        If ``True``, apply an ExB Doppler-shift correction in the (k, t)
        domain before taking the time FFT.  Default: ``False``.

    Returns
    -------
    k_norm : ndarray
        Normalised poloidal wavenumber axis ``ktheta * rho_i`` (one-sided).
    omega_norm : ndarray
        Normalised frequency axis ``omega * R/cs`` (zero-centred).
    power : ndarray, shape (Nk, Nt)
        Power spectral density.
    k_label : str
        Axis label for the wavenumber.
    omega_label : str
        Axis label for the frequency.
    slicetitle : str
        Slice description for the figure title.
    """
    twindow = [twindow] if not isinstance(twindow, list) else twindow
    norm   = simulation.normalization
    tscale = norm.dict['tscale']  # [s] per normalised time unit

    # First frame: discover grid dimensions and the reference arc-length grid
    field_RZ0, RR0, ZZ0, _ = polproj.get_projection(fieldName=fieldname,
                                                     timeFrame=twindow[0])
    NR = RR0.shape[0]
    # field_RZ and RR/ZZ may have different poloidal sizes (cell-centres vs edges)
    n_theta = field_RZ0.shape[1]  # authoritative length for the poloidal cut

    # Radial cut index closest to cut_x in the simulation x grid
    x_grid = simulation.geom_param.grids[0]  # physical x [m]
    icut   = int(np.argmin(np.abs(x_grid - cut_x)))

    def _arclength(RR, ZZ, i):
        """Cumulative arc-length along poloidal direction at row *i*.

        The returned array always has length *n_theta*, resampling RR/ZZ
        if their poloidal dimension differs (e.g. cell-edges vs cell-centres).
        """
        n_rz = RR.shape[1]
        dl = np.sqrt(np.diff(RR[i, :]) ** 2 + np.diff(ZZ[i, :]) ** 2)
        ll_rz = np.zeros(n_rz)
        ll_rz[1:] = np.cumsum(dl)
        if n_rz == n_theta:
            return ll_rz
        # Resample to n_theta points via linear interpolation on [0, 1]
        return np.interp(np.linspace(0.0, 1.0, n_theta),
                         np.linspace(0.0, 1.0, n_rz),
                         ll_rz)

    ll_ref     = _arclength(RR0, ZZ0, icut)
    L          = ll_ref[-1]
    ll_uniform = np.linspace(0.0, L, n_theta)

    Nt = len(twindow)
    phi_theta_t = np.zeros((n_theta, Nt))
    t_phys      = np.zeros(Nt)

    # Optional ExB correction bookkeeping
    if exb_correction:
        exb_vtheta_t = np.zeros(Nt)
        try:
            r0 = simulation.geom_param.r_x(cut_x)
            R0 = simulation.geom_param.R_x(cut_x)
            q0 = simulation.geom_param.qprofile_x(cut_x)
            pitch_factor = 1.0 / (1.0 + r0 / (R0 * q0))
        except Exception:
            pitch_factor = 1.0

    for it, tf in enumerate(twindow):
        field_RZ, RR, ZZ, t_norm_frame = polproj.get_projection(
            fieldName=fieldname, timeFrame=tf)
        t_phys[it] = t_norm_frame * tscale  # physical time [s]

        ll = _arclength(RR, ZZ, icut)
        phi_theta = field_RZ[icut, :]

        # Resample onto uniform arc-length grid to ensure correct wavenumbers
        phi_uniform = interp1d(ll, phi_theta, kind='linear',
                               bounds_error=False,
                               fill_value=(phi_theta[0], phi_theta[-1]))(ll_uniform)
        phi_theta_t[:, it] = phi_uniform

        if exb_correction:
            try:
                Bmag_RZ, _, _, _ = polproj.get_projection(fieldName='Bmag',
                                                          timeFrame=tf)
                # Radial arc-length for gradient computation
                drR = np.diff(RR, axis=0)
                drZ = np.diff(ZZ, axis=0)
                dr_arr = np.zeros_like(RR)
                dr_arr[1:, :] = np.sqrt(drR ** 2 + drZ ** 2)
                rr = np.cumsum(dr_arr, axis=0)
                # Mean poloidal ExB velocity at the cut
                ExB_vth = []
                for j in range(1, n_theta - 1):
                    Bj    = Bmag_RZ[icut, j]
                    dr_ij = rr[icut + 1, j] - rr[icut - 1, j]
                    if Bj != 0.0 and dr_ij != 0.0:
                        dphidr = (field_RZ[icut + 1, j] -
                                  field_RZ[icut - 1, j]) / dr_ij
                        ExB_vth.append(pitch_factor / Bj * dphidr)
                exb_vtheta_t[it] = np.mean(ExB_vth) if ExB_vth else 0.0
            except Exception:
                exb_vtheta_t[it] = 0.0

    # ------------------------------------------------------------------ #
    # 2D FFT with optional ExB Doppler correction                         #
    # ------------------------------------------------------------------ #
    dl_uniform = ll_uniform[1] - ll_uniform[0]   # uniform physical spacing [m]
    dt_phys_arr = np.mean(np.diff(t_phys))        # mean physical dt [s]

    win_t = np.kaiser(Nt, beta=8)
    phi_windowed = phi_theta_t * win_t[np.newaxis, :]

    # Spatial FFT (one-sided)
    phi_k_t = np.fft.rfft(phi_windowed, axis=0)

    if exb_correction:
        # Apply ExB Doppler correction: multiply by exp(i * k * (-v_ExB) * t)
        k_phys_vec = np.fft.rfftfreq(n_theta, d=dl_uniform) * 2.0 * np.pi  # rad/m
        kk, tt = np.meshgrid(k_phys_vec, t_phys, indexing='ij')
        vv     = np.ones_like(kk) * exb_vtheta_t[np.newaxis, :]
        phi_k_t = phi_k_t * np.exp(1j * kk * (-vv) * tt)

    # Time FFT + fftshift
    phi_k_omega = np.fft.fftshift(np.fft.fft(phi_k_t, axis=1), axes=1)

    # Axes
    k_phys  = np.fft.rfftfreq(n_theta, d=dl_uniform) * 2.0 * np.pi  # rad/m
    k_scale = norm.dict['kthetascale']   # 1/rho_i
    k_norm  = k_phys / k_scale           # ktheta * rho_i

    omega_phys = (np.fft.fftshift(np.fft.fftfreq(Nt, d=dt_phys_arr))
                  * 2.0 * np.pi)         # rad/s
    omega_scale = norm.dict['omegascale']  # R/cs [s]
    omega_norm  = omega_phys * omega_scale

    power = np.abs(phi_k_omega) ** 2
    # Drop the k=0 (DC) mode and normalise so the total power sums to 1
    k_norm = k_norm[1:]
    power  = power[1:]
    total  = power.sum()
    if total > 0:
        power = power / total

    # Axis labels
    ktheta_sym  = norm.dict['kthetasymbol']
    ktheta_unit = norm.dict['kthetaunits']
    k_label     = ktheta_sym + (' [' + ktheta_unit + ']') * (ktheta_unit != '')
    omega_sym   = norm.dict['omegasymbol']
    omega_unit  = norm.dict['omegaunits']
    omega_label = omega_sym + (' [' + omega_unit + ']') * (omega_unit != '')

    norm_dict  = simulation.normalization.dict
    x_norm_val = cut_x / norm_dict['xscale'] - norm_dict['xshift']
    slicetitle = norm_dict['xsymbol'] + ('=%.2f' % x_norm_val)

    return k_norm, omega_norm, power, k_label, omega_label, slicetitle
