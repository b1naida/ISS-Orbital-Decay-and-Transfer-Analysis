import numpy as np
import matplotlib.pyplot as plt
import requests
from sgp4.api import Satrec

MU = 398600.4418   # Earth's gravitational parameter [km^3/s^2]
RE = 6378.137      # Earth's equatorial radius [km]

URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=tle"

# Sample ISS TLE (used only if the download fails; its epoch is from 2019)
FALLBACK = (
    "ISS (ZARYA)",
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
    "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482",
)


def load_tle():
    try:
        resp = requests.get(URL, timeout=15)
        resp.raise_for_status()
        text = resp.text.strip().splitlines()
        name, l1, l2 = text[0].strip(), text[1].strip(), text[2].strip()
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            raise ValueError("response is not a valid TLE")
        print(f"[OK] Downloaded TLE from CelesTrak: {name}")
        return name, l1, l2
    except Exception as err:
        print(f"[WARN] Download failed ({err}). Using sample TLE.")
        return FALLBACK


def main():
    name, l1, l2 = load_tle()
    sat = Satrec.twoline2rv(l1, l2)

    # --- Orbital parameters from the TLE ---
    n = sat.no_kozai / 60.0                 # mean motion [rad/s]
    a = (MU / n**2) ** (1.0 / 3.0)          # semi-major axis [km]
    e = sat.ecco                            # eccentricity
    period = 2 * np.pi / n                  # orbital period [s]
    perigee_alt = a * (1 - e) - RE
    apogee_alt = a * (1 + e) - RE

    print(f"\n=== {name} ===")
    print(f"Semi-major axis : {a:10.2f} km")
    print(f"Eccentricity    : {e:10.6f}")
    print(f"Inclination     : {np.degrees(sat.inclo):10.3f} deg")
    print(f"Period          : {period / 60:10.2f} min")
    print(f"Perigee altitude: {perigee_alt:10.2f} km")
    print(f"Apogee altitude : {apogee_alt:10.2f} km")

    # --- Propagate one full orbit with SGP4 ---
    t = np.arange(0, period, 10.0)          # seconds since epoch, 10 s steps
    alt, v_sgp4 = [], []
    for ti in t:
        err, r, v = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF + ti / 86400.0)
        if err != 0:
            raise RuntimeError(f"SGP4 error code {err}")
        r_norm = np.linalg.norm(r)
        alt.append(r_norm - RE)             # spherical-Earth approximation
        v_sgp4.append(np.linalg.norm(v))
    alt, v_sgp4 = np.array(alt), np.array(v_sgp4)

    # --- Sanity check: does the result make physical sense? ---
    if not (350 < alt.mean() < 450):
        print(f"[WARN] Mean altitude {alt.mean():.0f} km is outside the expected "
              f"ISS range (350-450 km). Check the TLE.")

    # --- Our own Keplerian model: vis-viva equation ---
    # Uses the radius from SGP4 and the semi-major axis from the TLE mean motion.
    r_arr = alt + RE
    v_visviva = np.sqrt(MU * (2.0 / r_arr - 1.0 / a))

    # --- Difference between the two models (not an error against ground truth) ---
    dv = v_visviva - v_sgp4                 # [km/s], signed
    rel_err = np.abs(dv) / v_sgp4 * 100
    print("\nVis-viva vs SGP4 speed (two different models):")
    print(f"  SGP4 speed range     : {v_sgp4.min():.4f} - {v_sgp4.max():.4f} km/s")
    print(f"  Vis-viva speed range : {v_visviva.min():.4f} - {v_visviva.max():.4f} km/s")
    print(f"  Difference (vis-viva - SGP4): mean {dv.mean() * 1000:+.1f} m/s, "
          f"max |diff| {np.abs(dv).max() * 1000:.1f} m/s")
    print(f"  Relative difference : mean {rel_err.mean():.3f} %, max {rel_err.max():.3f} %")

    # --- Plots ---
    t_min = t / 60.0
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(t_min, alt)
    ax[0].set_xlabel("Time since epoch [min]")
    ax[0].set_ylabel("Altitude [km]")
    ax[0].set_title("Altitude over one orbit (SGP4)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t_min, v_sgp4, label="SGP4")
    ax[1].plot(t_min, v_visviva, "--", label="Vis-viva (own model)")
    ax[1].text(0.02, 0.04,
               f"vis-viva - SGP4: mean {dv.mean() * 1000:+.0f} m/s\n"
               f"max |diff| {np.abs(dv).max() * 1000:.0f} m/s",
               transform=ax[1].transAxes, fontsize=9, va="bottom")
    ax[1].set_xlabel("Time since epoch [min]")
    ax[1].set_ylabel("Speed [km/s]")
    ax[1].set_title("Speed: SGP4 vs vis-viva")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig("iss_week1.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()