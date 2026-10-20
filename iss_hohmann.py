try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        f"Missing package: {exc.name}\n"
        "Install it with the SAME Python you run this script with:\n"
        "    python -m pip install numpy scipy matplotlib"
    )

try:
    from scipy.integrate import solve_ivp
    HAVE_SCIPY = True
except ImportError:          # SciPy is optional: a built-in RK4 integrator is used instead
    HAVE_SCIPY = False

MU = 398600.4418          # Earth's gravitational parameter [km^3/s^2]
RE = 6378.137             # Earth's equatorial radius [km]
G0 = 9.80665              # standard gravity [m/s^2]

H1_KM = 420.0             # starting altitude (ISS-like, from week 2 data)
SIDEREAL_DAY_S = 86164.0905
INCLINATION_DEG = 51.64   # ISS inclination (from week 1)
ISP_S = [320, 450]        # specific impulse examples: storable / LOX-LH2


def geo_radius():
    """Radius of a circular orbit with a period of one sidereal day."""
    return (MU * (SIDEREAL_DAY_S / (2 * np.pi)) ** 2) ** (1.0 / 3.0)


def hohmann(r1, r2):
    """Coplanar Hohmann transfer between circular orbits of radius r1 -> r2 [km]."""
    a_t = (r1 + r2) / 2.0                          # transfer ellipse semi-major axis
    v1 = np.sqrt(MU / r1)                          # circular speed at r1
    v2 = np.sqrt(MU / r2)                          # circular speed at r2
    v_peri = np.sqrt(MU * (2 / r1 - 1 / a_t))      # transfer speed at r1 (vis-viva)
    v_apo = np.sqrt(MU * (2 / r2 - 1 / a_t))       # transfer speed at r2 (vis-viva)
    dv1 = v_peri - v1
    dv2 = v2 - v_apo
    t_transfer = np.pi * np.sqrt(a_t**3 / MU)      # half an ellipse period
    return dict(a_t=a_t, v1=v1, v2=v2, v_peri=v_peri, v_apo=v_apo,
                dv1=dv1, dv2=dv2, dv_total=dv1 + dv2, t_transfer=t_transfer)


def combined_burn(v_a, v_b, d_inc_deg):
    """Delta-v for changing speed v_a -> v_b and rotating the plane by d_inc (law of cosines)."""
    d = np.radians(d_inc_deg)
    return np.sqrt(v_a**2 + v_b**2 - 2 * v_a * v_b * np.cos(d))


def propellant_fraction(dv_kms, isp_s):
    """Tsiolkovsky: propellant mass / initial mass for a given delta-v."""
    return 1.0 - np.exp(-dv_kms * 1000.0 / (isp_s * G0))


def two_body(t, y):
    x, yy, vx, vy = y
    r3 = np.hypot(x, yy) ** 3
    return [vx, vy, -MU * x / r3, -MU * yy / r3]


def rk4(f, y0, t_end, dt=1.0):
    """Classic fixed-step Runge-Kutta 4 (fallback when SciPy is not installed)."""
    n = int(np.ceil(t_end / dt))
    h = t_end / n
    t = np.linspace(0.0, t_end, n + 1)
    y = np.empty((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        k1 = np.array(f(t[i], y[i]))
        k2 = np.array(f(t[i] + h / 2, y[i] + h / 2 * k1))
        k3 = np.array(f(t[i] + h / 2, y[i] + h / 2 * k2))
        k4 = np.array(f(t[i] + h, y[i] + h * k3))
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y.T


def integrate_transfer(y0, t_end):
    """Integrate the two-body ODE from 0 to t_end. Returns (t, states[4 x N], method name)."""
    if HAVE_SCIPY:
        t_eval = np.linspace(0.0, t_end, 2000)
        sol = solve_ivp(two_body, (0, t_end), y0, t_eval=t_eval, rtol=1e-11, atol=1e-11)
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        return sol.t, sol.y, "SciPy solve_ivp"
    print("[WARN] SciPy not found: using the built-in RK4 integrator (dt = 1 s).")
    print("       Install SciPy with:  python -m pip install scipy")
    t, y = rk4(two_body, y0, t_end, dt=1.0)
    return t, y, "built-in RK4"


def main():
    r1 = RE + H1_KM
    r2 = geo_radius()
    h = hohmann(r1, r2)

    print(f"Start orbit : r1 = {r1:.1f} km  (altitude {H1_KM:.0f} km)")
    print(f"Target orbit: r2 = {r2:.1f} km  (GEO altitude {r2 - RE:.0f} km)")
    print("\n=== Coplanar Hohmann transfer ===")
    print(f"  Burn 1 (at r1): {h['dv1']:.3f} km/s")
    print(f"  Burn 2 (at r2): {h['dv2']:.3f} km/s")
    print(f"  Total delta-v : {h['dv_total']:.3f} km/s")
    print(f"  Transfer time : {h['t_transfer'] / 3600:.2f} h")

    # --- Plane change combined with the second burn ---
    dv2_pc = combined_burn(h["v_apo"], h["v2"], INCLINATION_DEG)
    dv_total_pc = h["dv1"] + dv2_pc
    print(f"\n=== With {INCLINATION_DEG:.2f} deg plane change (combined at apogee) ===")
    print(f"  Burn 2 incl. plane change: {dv2_pc:.3f} km/s")
    print(f"  Total delta-v            : {dv_total_pc:.3f} km/s")

    # --- Rocket equation ---
    print("\n=== Propellant needed (fraction of the initial mass) ===")
    for isp in ISP_S:
        print(f"  Isp {isp:3d} s: coplanar {propellant_fraction(h['dv_total'], isp) * 100:5.1f} %"
              f"   with plane change {propellant_fraction(dv_total_pc, isp) * 100:5.1f} %")

    # --- Validation: numerical integration of the transfer orbit ---
    y0 = [r1, 0.0, 0.0, h["v_peri"]]               # start at perigee, moving along +y
    t_sol, y_sol, method = integrate_transfer(y0, h["t_transfer"])
    xf, yf, vxf, vyf = y_sol[:, -1]
    r_end = np.hypot(xf, yf)
    v_end = np.hypot(vxf, vyf)
    print(f"\n=== Validation: numerical integration of the transfer ellipse ({method}) ===")
    print(f"  Radius after t_transfer: {r_end:.3f} km   (target {r2:.3f} km, "
          f"error {abs(r_end - r2) * 1000:.2f} m)")
    print(f"  Speed  after t_transfer: {v_end:.5f} km/s (analytic {h['v_apo']:.5f} km/s, "
          f"error {abs(v_end - h['v_apo']) * 1000:.4f} m/s)")

    # --- Plots ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})

    th = np.linspace(0, 2 * np.pi, 400)
    ax[0].fill(RE * np.cos(th), RE * np.sin(th), color="lightblue", label="Earth")
    ax[0].plot(r1 * np.cos(th), r1 * np.sin(th), "b-", lw=1, label=f"Start orbit ({H1_KM:.0f} km)")
    ax[0].plot(r2 * np.cos(th), r2 * np.sin(th), "g-", lw=1, label="GEO")
    ax[0].plot(y_sol[0], y_sol[1], "r--", lw=1.5, label="Transfer ellipse (numerical)")
    ax[0].plot([r1], [0], "ko")
    ax[0].plot([-r2], [0], "ko")
    ax[0].annotate(f"Burn 1\n{h['dv1']:.2f} km/s", (r1, 0), (r1 + 6000, 6000), fontsize=9,
                   arrowprops=dict(arrowstyle="->"))
    ax[0].annotate(f"Burn 2\n{h['dv2']:.2f} km/s", (-r2, 0), (-r2 + 6000, 5000), fontsize=9,
                   arrowprops=dict(arrowstyle="->"))
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x [km]")
    ax[0].set_ylabel("y [km]")
    ax[0].set_title("Hohmann transfer: ISS altitude -> GEO")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(alpha=0.3)

    labels = ["Burn 1", "Burn 2\n(coplanar)", "Burn 2 incl.\nplane change"]
    values = [h["dv1"], h["dv2"], dv2_pc]
    ax[1].bar(labels, values, color=["tab:blue", "tab:green", "tab:red"])
    for i, v in enumerate(values):
        ax[1].text(i, v + 0.03, f"{v:.2f}", ha="center")
    ax[1].set_ylabel("Delta-v [km/s]")
    ax[1].set_title("Delta-v budget")
    ax[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("hohmann_week3.png", dpi=150)
    print("\nSaved plot: hohmann_week3.png")
    plt.show()


if __name__ == "__main__":
    main()