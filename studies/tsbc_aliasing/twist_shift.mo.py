import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")

# Cell: utility functions — defines the q-profile and low-pass filter used in the notebook.
@app.cell
def _():
    import numpy as np

    def q_func(r, r0, q0, nshear):
        return q0 * pow(r/r0, nshear)
    
    return q_func, np



# Cell: introductory markdown — explains the twist-and-shift boundary condition and what the notebook does.
# @app.cell
# def _(mo):
#     mo.md(
#         r"""
#         # Twist-and-Shift Boundary Conditions Explorer

#         This notebook lets you interactively explore twist-and-shift boundary conditions
#         used in flux-tube gyrokinetic simulations.

#         Given a field $\phi(x, y)$ with a single $(k_x, k_y)$ wave, this notebook applies 
#         the twist-and-shift map directly in real space, using the periodic remapping

#         $$y \to y + 2\pi r_0/q_0 q(x).$$

#         It displays the original field, the twist-shifted field on the periodic
#         grid, their Fourier-space spectra, and the associated $q(x)$ profile.
#         """
#     )
#     return


# Cell: imports — loads marimo (reactive UI) and numpy.
@app.cell
def _():
    import marimo as mo
    return mo


# Cell: imports — loads matplotlib for all plotting.
@app.cell
def _():
    import matplotlib.pyplot as plt
    return plt


# Cell: grid-size sliders — lets the user choose the number of grid points Nx and Ny.
@app.cell
def _(mo):
    Nx_slider = mo.ui.slider(value=32, start=4, stop=256, step=4, label="Nx (grid points in x)")
    Ny_slider = mo.ui.slider(value=32, start=4, stop=256, step=4, label="Ny (grid points in y)")
    return Nx_slider, Ny_slider

# Cell: wave-parameter sliders — selects the field mode (plane wave or Gaussian packet) and
#   the integer mode numbers kx_mode, ky_mode, plus the Gaussian envelope widths sigma_x/y.
@app.cell
def _(mo):
    # --- Wave parameters ---
    # field_mode: "single" is a pure cosine plane wave; "gaussian" is a
    #   localised wave-packet (Gaussian envelope × cosine carrier).
    # kx_mode / ky_mode: integer mode numbers; the physical wave-numbers
    #   are kx = kx_mode * 2π/Lx and ky = ky_mode * 2π/Ly.
    # sigma_x / sigma_y: half-widths of the Gaussian envelope (in box units);
    #   only used when field_mode == "gaussian".
    field_mode = mo.ui.dropdown(options=["single", "gaussian"], value="single", label="Field mode")
    kx_mode_slider = mo.ui.slider(value=1, start=0, stop=8, step=1, label="kx mode number (integer)")
    ky_mode_slider = mo.ui.slider(value=1, start=0, stop=8, step=1, label="ky mode number (integer)")
    sigma_x_slider = mo.ui.slider(value=0.1, start=0.05, stop=2.0, step=0.01, label="sigma_x (plane units)")
    sigma_y_slider = mo.ui.slider(value=1.0, start=0.1, stop=6.0, step=0.01, label="sigma_y (plane units)")
    return field_mode, kx_mode_slider, ky_mode_slider, sigma_x_slider, sigma_y_slider

# Cell: magnetic-shear slider — sets the normalised shear ŝ that governs the twist-and-shift offset.
@app.cell
def _(mo):
    shat_slider = mo.ui.slider(value=0.5, start=0.0, stop=5.0, step=0.5, label="ŝ (magnetic shear)")
    q0_slider = mo.ui.slider(value=1.0, start=1.0, stop=5.0, step=0.1, label="q0 (safety factor)")
    return shat_slider, q0_slider

# Cell: visualisation-control widgets — colormap picker and oversampling factors for the shift.
@app.cell
def _(mo):
    cmap_dropdown = mo.ui.dropdown(options=["seismic", "viridis", "twilight"], value="seismic", label="Colormap")
    nint_x_slider = mo.ui.slider(value=1, start=1, stop=8, step=1, label="Oversampling x (nint_x)")
    nint_y_slider = mo.ui.slider(value=1, start=1, stop=8, step=1, label="Oversampling y (nint_y)")
    return cmap_dropdown, nint_x_slider, nint_y_slider


# Cell: core computation — builds the grid, constructs the test field phi(x,y), applies the
#   twist-and-shift via spectral y-shifts, and optionally low-pass filters both fields.
@app.cell
def _(field_mode,nint_x_slider,nint_y_slider,Nx_slider,Ny_slider,kx_mode_slider,
      ky_mode_slider,np,shat_slider,q0_slider,sigma_x_slider,sigma_y_slider,q_func):
    
    # --- Unpack slider values into plain variables ---
    Nx = Nx_slider.value
    Ny = Ny_slider.value
    x0 = 2.0   # reference radial position (centre of the x-domain)
    Lx = 1.0   # radial box size
    A = 1.0    # target amplitude after normalisation
    kx_mode = kx_mode_slider.value
    ky_mode = ky_mode_slider.value
    shat = shat_slider.value
    q0 = q0_slider.value   # safety factor at the reference surface x0
    Cy = x0/q0
    Ly = 2*np.pi*Cy

    # --- Build the 2-D spatial grid ---
    dx = Lx / Nx
    dy = Ly / Ny
    x = np.linspace(x0, x0 + Lx - dx, Nx)
    y = np.linspace(-Ly / 2, Ly / 2 - dy, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")  # shape (Nx, Ny)

    # Fundamental wave-numbers (smallest non-zero k on this grid)
    dkx = 2.0 * np.pi / Lx
    dky = 2.0 * np.pi / Ly

    # Physical wave-numbers of the chosen mode
    kx_val = kx_mode * dkx
    ky_val = ky_mode * dky

    # Local radial coordinate measured from the reference surface at x0.
    # Used so that the shear shift is zero at x = x0.
    x_local = x - x0

    # --- safety-factor profile ---
    q_profile = q_func(x, x0, q0, shat)

    # --- Construct the original 2-D test field phi(x, y) ---
    if field_mode.value == "single":
        # Pure plane wave: a single Fourier mode filling the whole domain.
        phi = np.cos(kx_val * X + ky_val * Y)
    else:
        # Gaussian wave-packet: a localised envelope modulating the carrier.
        # This is useful for visualising how a spatially-confined perturbation
        # is remapped by the twist-and-shift.
        x_center = x0
        y_center = 0.0
        sigma_x = sigma_x_slider.value
        sigma_y = sigma_y_slider.value * Cy

        # Gaussian envelope centred at (x_center, y_center)
        envelope = np.exp(
            -0.5 * ((X - x_center) / sigma_x) ** 2
            -0.5 * ((Y - y_center) / sigma_y) ** 2
        )
        # Cosine carrier wave (the wave-like oscillation inside the packet)
        carrier = np.cos(kx_val * (X - x_center) + ky_val * (Y - y_center))
        phi = envelope * carrier

    # Normalize to unit amplitude so the colour scale is the same for both modes.
    phi_max = np.max(np.abs(phi))
    if phi_max > 0:
        phi = A * phi / phi_max

    # Coarse-grid y-shift (kept for diagnostics)
    delta_y = 2.0 * np.pi * Cy * q_profile  # shape (Nx,)

    # --- 1. Upsample phi to fine grid (separable: periodic in y, linear in x) ---
    nint_x = nint_x_slider.value
    nint_y = nint_y_slider.value
    Nx_fine = Nx * nint_x
    Ny_fine = Ny * nint_y
    dx_fine = Lx / Nx_fine
    dy_fine = Ly / Ny_fine
    x_fine = np.linspace(x0, x0 + Lx - dx_fine, Nx_fine)
    y_fine = np.linspace(-Ly / 2, Ly / 2 - dy_fine, Ny_fine)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing="ij")

    phi_fine_y = np.zeros((Nx, Ny_fine))
    for i in range(Nx):
        phi_fine_y[i, :] = np.interp(y_fine, y, phi[i, :], period=Ly)
    phi_fine = np.zeros((Nx_fine, Ny_fine))
    for j in range(Ny_fine):
        phi_fine[:, j] = np.interp(x_fine, x, phi_fine_y[:, j])

    # --- 2. Apply twist-and-shift on the fine grid ---
    q_profile_fine = q_func(x_fine, x0, q0, shat)
    delta_y_fine = 2.0 * np.pi * Cy * q_profile_fine
    Y_shifted_fine = Y_fine + delta_y_fine[:, np.newaxis]
    Y_shifted_wrapped_fine = (Y_shifted_fine + Ly / 2) % Ly - Ly / 2

    phi_shifted_fine = np.zeros_like(phi_fine)
    for i in range(Nx_fine):
        phi_shifted_fine[i, :] = np.interp(
            Y_shifted_wrapped_fine[i, :], y_fine, phi_fine[i, :], period=Ly
        )

    # --- 3. Downsample shifted field back to original grid ---
    phi_shifted_coarse_y = np.zeros((Nx_fine, Ny))
    for i in range(Nx_fine):
        phi_shifted_coarse_y[i, :] = np.interp(y, y_fine, phi_shifted_fine[i, :], period=Ly)
    phi_shifted = np.zeros((Nx, Ny))
    for j in range(Ny):
        phi_shifted[:, j] = np.interp(x, x_fine, phi_shifted_coarse_y[:, j])

    return (A, x0, Lx, Ly, Nx, Ny, X, Y, delta_y, dky, dx, dy,
            field_mode, kx_val, ky_val, phi, phi_shifted,
            q_profile, shat, sigma_x_slider, sigma_y_slider, x, x_local, y)

# Cell: real-space field plots — side-by-side pcolormesh of the original and twist-shifted phi(x,y).
@app.cell
def _( X, Y, cmap_dropdown, mo, np, phi, phi_shifted, plt):
    # Side-by-side pcolormesh plots of the original and/or twist-shifted field.
    # Both panels share the same colour scale (vmin=-vmax, vmax=vmax) so that
    # amplitude differences are immediately visible.
    _cmap = cmap_dropdown.value
    _vmax = np.max(np.abs(phi))

    # Build a list of (title, data) pairs for whichever panels are enabled
    _panels = []
    _panels.append(("φ(x,y)", phi))
    _panels.append(("φ(x,y+S(x))", phi_shifted))

    _n = len(_panels)
    if _n > 0:
        _ncols = min(_n, 4)
        _nrows = int(np.ceil(_n / _ncols))
        _fig, _axes = plt.subplots(
            _nrows, _ncols, figsize=(5 * _ncols, 4.5 * _nrows), squeeze=False
        )
        for _i, (_title, _data) in enumerate(_panels):
            _r, _c = divmod(_i, _ncols)
            _ax = _axes[_r][_c]
            _vm = _vmax if "err" not in _title.lower() else max(np.max(np.abs(_data)), 1e-10)
            _im = _ax.pcolormesh(
                X, Y, _data, cmap=_cmap, vmin=-_vm, vmax=_vm, shading="auto"
            )
            _ax.set_title(_title)
            _ax.set_xlabel("x")
            _ax.set_ylabel("y")
            plt.colorbar(_im, ax=_ax, shrink=0.8)
        # Hide unused axes
        for _i in range(_n, _nrows * _ncols):
            _r, _c = divmod(_i, _ncols)
            _axes[_r][_c].set_visible(False)
        plt.tight_layout()
        fig_realspace = plt.gcf()
        plt.close(fig_realspace)
    else:
        fig_realspace = mo.md("*Enable at least one panel above.*")
    return fig_realspace,


# Cell: 2-D Fourier spectra — pcolormesh of |φ̂(kx,ky)| for the original and shifted fields,
#   showing how the twist-and-shift mixes Fourier modes.
@app.cell
def _(Lx, Ly, Nx, Ny, np, phi, phi_shifted, plt):
    _phi_hat = np.fft.fftshift(np.fft.fft2(phi)) / (Nx * Ny)
    _phi_rs_hat = np.fft.fftshift(np.fft.fft2(phi_shifted)) / (Nx * Ny)

    # Build centred kx/ky axes (fftshift reorders from [0,+k,...,-k] to [-k,...,0,...,+k])
    _kx_axis = np.fft.fftshift(np.fft.fftfreq(Nx, d=Lx / Nx)) * 2 * np.pi
    _ky_axis = np.fft.fftshift(np.fft.fftfreq(Ny, d=Ly / Ny)) * 2 * np.pi
    _KX, _KY = np.meshgrid(_kx_axis, _ky_axis, indexing="ij")

    _spectra = [
        ("Original |φ̂|", np.abs(_phi_hat)),
        ("Shifted |φ̂|", np.abs(_phi_rs_hat)),
    ]

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=False)
    _axes = _axes[0]
    for _i, (_title, _data) in enumerate(_spectra):
        _vm = np.max(_data) or 1e-10
        _im = _axes[_i].pcolormesh(
            _KX, _KY, _data, cmap="hot", vmin=0, vmax=_vm, shading="auto"
        )
        _axes[_i].set_title(_title)
        _axes[_i].set_xlabel("kx")
        _axes[_i].set_ylabel("ky")
        plt.colorbar(_im, ax=_axes[_i], shrink=0.8)
    plt.tight_layout()
    fig_fourier = plt.gcf()
    plt.close(fig_fourier)
    return fig_fourier,


# Cell: safety-factor and shear profiles — plots q(x) and the local normalised shear ŝ(x),
#   letting the user verify how the linear q-model behaves across the radial box.
@app.cell
def _(np, plt, q_profile, shat, x_local):
    _fig4, (_ax_q, _ax_s) = plt.subplots(1, 2, figsize=(10, 3.5))

    _ax_q.plot(x_local, q_profile, "k-", lw=2)
    _ax_q.set_xlabel("x - x0")
    _ax_q.set_ylabel("q(x)")
    _ax_q.set_title("Safety factor profile")
    _ax_q.grid(True, alpha=0.3)

    # Shear profile: shat_local = (x_local/q) * dq/dx_local
    # For linear q: dq/dx_local = shat (parameter)
    # Guard against division by zero if q ever passes through zero.
    _shat_local = x_local/q_profile * np.gradient(q_profile, x_local)
    _ax_s.plot(x_local, _shat_local, "b-", lw=2, label="ŝ_local = (x/q) dq/dx")
    _ax_s.axhline(shat, color="r", ls="--", label=f"ŝ parameter = {shat:.2f}")
    _ax_s.set_xlabel("x - x0")
    _ax_s.set_ylabel("ŝ(x)")
    _ax_s.set_title("Local magnetic shear")
    _ax_s.legend()
    _ax_s.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_profiles = plt.gcf()
    plt.close(fig_profiles)
    return fig_profiles,

# Cell: main layout — parameters on the left, plots on the right.
@app.cell
def _(
    Nx_slider, Ny_slider,
    field_mode, kx_mode_slider, ky_mode_slider, sigma_x_slider, sigma_y_slider,
    shat_slider, q0_slider,
    cmap_dropdown, nint_x_slider, nint_y_slider,
    fig_realspace, fig_fourier, fig_profiles,
    mo,
):
    _left = mo.vstack([
        mo.hstack([Nx_slider]),
        mo.hstack([Ny_slider]),
        mo.hstack([field_mode]),
        mo.hstack([kx_mode_slider]),
        mo.hstack([ky_mode_slider]),
        mo.hstack([sigma_x_slider]),
        mo.hstack([sigma_y_slider]),
        mo.hstack([q0_slider]),
        mo.hstack([shat_slider]),
        mo.hstack([cmap_dropdown]),
        mo.hstack([nint_x_slider]),
        mo.hstack([nint_y_slider]),
    ])
    _right = mo.vstack([
        mo.md("## Profiles & Diagnostics"),
        fig_profiles,
        mo.md("## Real-Space Fields"),
        fig_realspace,
        mo.md("## Fourier Space"),
        fig_fourier,
    ])
    mo.Html(f"""
    <div style="display:flex;height:calc(100vh - 80px);gap:16px;overflow:hidden;">
      <div style="width:300px;flex-shrink:0;overflow-y:auto;border-right:1px solid #e0e0e0;padding-right:12px;box-sizing:border-box;">
        {_left.text}
      </div>
      <div style="flex:1;overflow-y:auto;min-width:0;">
        {_right.text}
      </div>
    </div>
    """)


if __name__ == "__main__":
    app.run()
