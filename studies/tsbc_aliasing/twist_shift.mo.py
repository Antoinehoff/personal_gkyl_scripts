import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Twist-and-Shift Boundary Conditions Explorer

        This notebook lets you interactively explore twist-and-shift boundary conditions
        used in flux-tube gyrokinetic simulations.

        Given a field $\phi(x, y)$ with a single $(k_x, k_y)$ wave, the twist-and-shift
        boundary condition at $z = \pm \pi$ connects the field-line-following coordinate
        via

        $$\phi(x, y, z + 2\pi) = \phi\!\left(x,\; y + 2\pi \hat{s} x,\; z\right)$$

        where $\hat{s} = (r/q)\,dq/dr$ is the magnetic shear.

        This notebook applies the twist-and-shift map directly in real space,
        using the periodic remapping

        $$y \to y + 2\pi \hat{s} x.$$

        It displays the original field, the twist-shifted field on the periodic
        grid, their Fourier-space spectra, and the associated $q(x)$ profile.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _():
    import matplotlib.pyplot as plt
    return plt


@app.cell
def _(mo):
    mo.md("## Parameters")
    return


@app.cell
def _(mo):
    # --- Grid parameters ---
    Nx_slider = mo.ui.number(value=32, start=4, stop=256, step=4, label="Nx (grid points in x)")
    Ny_slider = mo.ui.number(value=32, start=4, stop=256, step=4, label="Ny (grid points in y)")
    mo.hstack([Nx_slider, Ny_slider])
    return Nx_slider, Ny_slider

@app.cell
def _(mo):
    # --- Wave parameters ---
    field_mode = mo.ui.dropdown(
        options=["single", "gaussian"],
        value="gaussian",
        label="Field mode",
    )
    kx_mode_slider = mo.ui.number(value=1, start=0, stop=50, step=1, label="kx mode number (integer)")
    ky_mode_slider = mo.ui.number(value=1, start=0, stop=50, step=1, label="ky mode number (integer)")
    sigma_x_slider = mo.ui.number(value=0.05, start=0.01, stop=2.0, step=0.01, label="sigma_x (plane units)")
    sigma_y_slider = mo.ui.number(value=0.05, start=0.01, stop=2.0, step=0.01, label="sigma_y (plane units)")
    return field_mode, kx_mode_slider, ky_mode_slider, sigma_x_slider, sigma_y_slider


@app.cell
def _(field_mode, kx_mode_slider, ky_mode_slider, mo, sigma_x_slider, sigma_y_slider):
    mo.vstack(
        [
            mo.hstack([field_mode, kx_mode_slider, ky_mode_slider]),
            mo.hstack([sigma_x_slider, sigma_y_slider]),
        ]
    )
    return


@app.cell
def _(mo):
    # --- shear parameter ---
    shat_slider = mo.ui.number(value=0.5, start=-10.0, stop=10.0, step=0.1, label="ŝ (magnetic shear)")
    mo.hstack([shat_slider])
    return shat_slider


@app.cell
def _(mo):
    # --- Visualisation ---
    cmap_dropdown = mo.ui.dropdown(
        options=["seismic", "viridis", "twilight"],
        value="seismic",
        label="Colormap",
    )
    show_original = mo.ui.checkbox(value=True, label="Show original field")
    show_shifted = mo.ui.checkbox(value=True, label="Show shifted field")
    apply_lowpass = mo.ui.checkbox(value=False, label="Apply Laplacian low-pass")
    lowpass_alpha = mo.ui.number(
        value=0.05,
        start=0.0,
        stop=10.0,
        step=0.01,
        label="Low-pass strength α",
    )
    lowpass_steps = mo.ui.number(
        value=4,
        start=1,
        stop=100,
        step=1,
        label="Low-pass iterations",
    )
    return apply_lowpass, cmap_dropdown, lowpass_alpha, lowpass_steps, show_original, show_shifted


@app.cell
def _(
    apply_lowpass,
    cmap_dropdown,
    lowpass_alpha,
    lowpass_steps,
    mo,
    show_original,
    show_shifted,
):
    mo.vstack(
        [
            mo.hstack([cmap_dropdown, show_original, show_shifted, apply_lowpass]),
            mo.hstack([lowpass_alpha, lowpass_steps]),
        ]
    )
    return


@app.cell
def _(
    apply_lowpass,
    field_mode,
    lowpass_alpha,
    lowpass_steps,
    Nx_slider,
    Ny_slider,
    kx_mode_slider,
    ky_mode_slider,
    np,
    shat_slider,
    sigma_x_slider,
    sigma_y_slider,
):
    # --- Unpack parameters ---
    Nx = Nx_slider.value
    Ny = Ny_slider.value
    x0 = 2.0
    Lx = 1.0
    Ly = 1.0
    kx_mode = kx_mode_slider.value
    ky_mode = ky_mode_slider.value
    A = 1.0
    q0 = 1.5
    shat = shat_slider.value

    # --- Grids ---
    dx = Lx / Nx
    dy = Ly / Ny
    x = np.linspace(x0 - Lx / 2, x0 + Lx / 2 - dx, Nx)
    y = np.linspace(-Ly / 2, Ly / 2 - dy, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")  # shape (Nx, Ny)

    # Fundamental wave-numbers
    dkx = 2.0 * np.pi / Lx
    dky = 2.0 * np.pi / Ly

    kx_val = kx_mode * dkx
    ky_val = ky_mode * dky

    # Local radial coordinate around the reference surface at x0.
    x_local = x - x0

    # --- q-profile: q(x_local) = q0 + shat * x_local ---
    q_profile = q0 + shat * x_local

    # --- Original field ---
    if field_mode.value == "single":
        phi = np.cos(kx_val * X + ky_val * Y)
    else:
        # Gaussian packet centered in the x-y plane with carrier (kx, ky).
        x_center = x0
        y_center = 0.0
        sigma_x = sigma_x_slider.value
        sigma_y = sigma_y_slider.value

        envelope = np.exp(
            -0.5 * ((X - x_center) / sigma_x) ** 2
            -0.5 * ((Y - y_center) / sigma_y) ** 2
        )
        carrier = np.cos(kx_val * (X - x_center) + ky_val * (Y - y_center))
        phi = envelope * carrier

    # Normalize so both modes share the same max amplitude A.
    phi_max = np.max(np.abs(phi))
    if phi_max > 0:
        phi = A * phi / phi_max

    # =========================================================
    # Real-space twist-and-shift:  y -> y + 2*pi*shat*x
    # =========================================================
    delta_y = 2.0 * np.pi * shat * x_local  # shape (Nx,)
    Y_shifted = Y + delta_y[:, np.newaxis]
    # Wrap Y into the periodic domain (diagnostic only)
    Y_shifted_wrapped = (Y_shifted + Ly / 2) % Ly - Ly / 2

    # Apply periodic y-shift row-wise using Fourier phase shifts.
    ky_fft = np.fft.fftfreq(Ny, d=dy) * 2.0 * np.pi
    phi_realspace = np.zeros_like(phi)
    for i in range(Nx):
        row_hat = np.fft.fft(phi[i, :])
        phase = np.exp(1j * ky_fft * delta_y[i])
        phi_realspace[i, :] = np.real(np.fft.ifft(row_hat * phase))

    # Optional low-pass filtering in x-y using the Laplacian operator.
    def laplacian_lowpass(field, alpha, steps):
        out = field.copy()
        for _ in range(steps):
            lap = (
                np.roll(out, 1, axis=0)
                + np.roll(out, -1, axis=0)
                + np.roll(out, 1, axis=1)
                + np.roll(out, -1, axis=1)
                - 4.0 * out
            )
            out = out + alpha * lap
        return out

    if apply_lowpass.value:
        alpha = lowpass_alpha.value
        steps = int(lowpass_steps.value)
        phi_used = laplacian_lowpass(phi, alpha, steps)
        phi_realspace_used = laplacian_lowpass(phi_realspace, alpha, steps)
    else:
        alpha = 0.0
        steps = 0
        phi_used = phi
        phi_realspace_used = phi_realspace
        
    phi = phi_used
    phi_realspace = phi_realspace_used
    
    return (
        A,
        Lx,
        Ly,
        Nx,
        Ny,
        X,
        Y,
        Y_shifted,
        Y_shifted_wrapped,
        delta_y,
        dky,
        dx,
        dy,
        field_mode,
        alpha,
        kx_val,
        ky_val,
        phi_used,
        phi_realspace_used,
        steps,
        q_profile,
        shat,
        sigma_x_slider,
        sigma_y_slider,
        x,
        x_local,
        y,
    )


@app.cell
def _(
    alpha,
    dky,
    delta_y,
    field_mode,
    ky_val,
    mo,
    np,
    shat,
    sigma_x_slider,
    sigma_y_slider,
    steps,
):
    _info = f"""
    ## Twist-and-Shift Info

    | Quantity | Value |
    |---|---|
    | Field mode | {field_mode.value} |
    | Magnetic shear ŝ | {shat:.3f} |
    | ky used | {ky_val:.4f} |
    | sigma_x (plane units) | {sigma_x_slider.value:.3f} |
    | sigma_y (plane units) | {sigma_y_slider.value:.3f} |
    | low-pass α | {alpha:.3f} |
    | low-pass steps | {steps} |
    | dky (grid spacing) | {dky:.4f} |
    | Min y-shift | {np.min(delta_y):.4f} |
    | Max y-shift | {np.max(delta_y):.4f} |
    """
    mo.md(_info)
    return


@app.cell
def _(mo):
    mo.md("## Real-Space Fields")
    return


@app.cell
def _(
    x0,
    Lx,
    Ly,
    X,
    Y,
    cmap_dropdown,
    mo,
    np,
    phi,
    phi_realspace,
    plt,
    show_original,
    show_shifted,
):
    _cmap = cmap_dropdown.value
    _vmax = np.max(np.abs(phi))

    # Count visible panels
    _panels = []
    if show_original.value:
        _panels.append(("Original φ(x,y)", phi))
    if show_shifted.value:
        _panels.append(("Real-space (wrap y)", phi_realspace))

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
            _ax.set_xlim(x0 - Lx / 2, x0 + Lx / 2)
            _ax.set_ylim(-Ly / 2, Ly / 2)
            plt.colorbar(_im, ax=_ax, shrink=0.8)
        # Hide unused axes
        for _i in range(_n, _nrows * _ncols):
            _r, _c = divmod(_i, _ncols)
            _axes[_r][_c].set_visible(False)
        plt.tight_layout()
        mo.output.replace(plt.gcf())
    else:
        mo.md("*Enable at least one panel above.*")
    return (plt,)


@app.cell
def _(mo):
    mo.md("## Fourier Space Analysis")
    return


@app.cell
def _(Lx, Ly, Nx, Ny, mo, np, phi, phi_realspace, plt):
    _phi_hat = np.fft.fftshift(np.fft.fft2(phi)) / (Nx * Ny)
    _phi_rs_hat = np.fft.fftshift(np.fft.fft2(phi_realspace)) / (Nx * Ny)

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
    mo.output.replace(plt.gcf())
    return


@app.cell
def _(mo):
    mo.md("## 1D Fourier Spectra")
    return


@app.cell
def _(Lx, Ly, Nx, Ny, mo, np, phi, phi_realspace, plt):
    _norm = Nx * Ny
    _phi_hat_1d = np.fft.fftshift(np.fft.fft2(phi)) / _norm
    _phi_rs_hat_1d = np.fft.fftshift(np.fft.fft2(phi_realspace)) / _norm

    _kx_ax = np.fft.fftshift(np.fft.fftfreq(Nx, d=Lx / Nx)) * 2 * np.pi
    _ky_ax = np.fft.fftshift(np.fft.fftfreq(Ny, d=Ly / Ny)) * 2 * np.pi

    _P_orig_kx = np.sum(np.abs(_phi_hat_1d) ** 2, axis=1)
    _P_shift_kx = np.sum(np.abs(_phi_rs_hat_1d) ** 2, axis=1)
    _P_orig_ky = np.sum(np.abs(_phi_hat_1d) ** 2, axis=0)
    _P_shift_ky = np.sum(np.abs(_phi_rs_hat_1d) ** 2, axis=0)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    _ax1.semilogy(_kx_ax, _P_orig_kx, "k-o", ms=3, label="Original")
    _ax1.semilogy(_kx_ax, _P_shift_kx, "r--s", ms=3, label="Shifted")
    _ax1.set_xlabel("kx")
    _ax1.set_ylabel("Σ_ky |φ̂|²")
    _ax1.set_title("Power spectrum vs kx")
    _ax1.legend(fontsize=8)
    _ax1.grid(True, alpha=0.3)

    _ax2.semilogy(_ky_ax, _P_orig_ky, "k-o", ms=3, label="Original")
    _ax2.semilogy(_ky_ax, _P_shift_ky, "r--s", ms=3, label="Shifted")
    _ax2.set_xlabel("ky")
    _ax2.set_ylabel("Σ_kx |φ̂|²")
    _ax2.set_title("Power spectrum vs ky")
    _ax2.legend(fontsize=8)
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    mo.output.replace(plt.gcf())
    return


@app.cell
def _(mo, np, plt, q_profile, shat, x_local):
    _fig4, (_ax_q, _ax_s) = plt.subplots(1, 2, figsize=(10, 3.5))

    _ax_q.plot(x_local, q_profile, "k-", lw=2)
    _ax_q.set_xlabel("x - x0")
    _ax_q.set_ylabel("q(x)")
    _ax_q.set_title("Safety factor profile")
    _ax_q.grid(True, alpha=0.3)

    # Shear profile: shat_local = (x_local/q) * dq/dx_local
    # For linear q: dq/dx_local = shat (parameter)
    _shat_local = np.where(
        np.abs(q_profile) > 1e-10,
        shat * x_local / q_profile,
        0.0,
    )
    _ax_s.plot(x_local, _shat_local, "b-", lw=2, label="ŝ_local = (x/q) dq/dx")
    _ax_s.axhline(shat, color="r", ls="--", label=f"ŝ parameter = {shat:.2f}")
    _ax_s.set_xlabel("x - x0")
    _ax_s.set_ylabel("ŝ(x)")
    _ax_s.set_title("Local magnetic shear")
    _ax_s.legend()
    _ax_s.grid(True, alpha=0.3)

    plt.tight_layout()
    mo.output.replace(plt.gcf())
    return


@app.cell
def _(delta_y, dky, mo, np):
    _info2 = f"""
    ## Shift Diagnostics

    - **y-shift range**: [{np.min(delta_y):.3f}, {np.max(delta_y):.3f}]
    - **dky**: {dky:.4f}

    > **Tip**: Larger values of $\\hat{{s}}$ or $k_y$ increase the poloidal remapping
    > $y \\to y + 2\\pi \\hat{{s}} x$ across the radial extent of the box.
    """
    mo.md(_info2)
    return


if __name__ == "__main__":
    app.run()
