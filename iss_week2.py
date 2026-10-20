import csv
import getpass
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

# ============================================================
# 1. SETTINGS
# ============================================================

MU = 398600.4418
RE = 6378.137

NORAD_ID = 25544
DAYS = 730

# Reboost detection
WINDOW = 3
REBOOST_THRESHOLD_KM = 0.3

# Merge candidates closer than this interval
MIN_GAP_DAYS = 7

# Outlier removal
OUTLIER_WINDOW = 7
OUTLIER_KM = 0.3
REMOVE_OUTLIERS = True

# Optional histogram
SHOW_HIST = False

# Validation settings
VALIDATION_WINDOW = 3

# Save everything next to this Python script
OUTPUT_DIR = Path(__file__).resolve().parent

CACHE = OUTPUT_DIR / "iss_history.json"
REMOVED_CSV = OUTPUT_DIR / "removed_points.csv"
PLOT_FILE = OUTPUT_DIR / "iss_week2.png"
VALIDATION_CSV = OUTPUT_DIR / "reboost_validation.csv"
ALTITUDE_CSV = OUTPUT_DIR / "iss_altitude_data.csv"

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"

QUERY_URL = (
    "https://www.space-track.org/basicspacedata/query/class/gp_history/"
    f"NORAD_CAT_ID/{NORAD_ID}/EPOCH/>now-{DAYS}/"
    "orderby/EPOCH%20asc/format/json"
)


# ============================================================
# 2. DATA DOWNLOAD
# ============================================================

def download():
    """Download ISS GP history from Space-Track."""

    user = (
        os.environ.get("SPACETRACK_USER")
        or input("Space-Track email: ")
    )
    password = (
        os.environ.get("SPACETRACK_PASS")
        or getpass.getpass("Space-Track password: ")
    )

    with requests.Session() as session:
        response = session.post(
            LOGIN_URL,
            data={"identity": user, "password": password},
            timeout=30,
        )
        response.raise_for_status()

        if "Failed" in response.text:
            raise RuntimeError(
                "Space-Track login failed. Check your credentials."
            )

        print("[OK] Logged in. Downloading ISS history...")

        response = session.get(QUERY_URL, timeout=180)
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, list):
        raise ValueError("Unexpected response from Space-Track.")

    CACHE.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    print(f"[OK] Cached {len(data)} records: {CACHE.resolve()}")
    return data


def load_data():
    """Load cached history, or download it if unavailable."""

    if CACHE.exists():
        print(f"[INFO] Using cache: {CACHE.resolve()}")
        return json.loads(CACHE.read_text(encoding="utf-8"))

    return download()


# ============================================================
# 3. OUTLIER DETECTION
# ============================================================

def outlier_mask(alt, window=7, threshold=0.5):
    """
    Identify points that differ from their local rolling median.
    True = retain; False = flag as an outlier.
    """

    if window < 1 or window % 2 == 0:
        raise ValueError("OUTLIER_WINDOW must be a positive odd number.")

    half = window // 2
    padded = np.pad(alt, half, mode="edge")

    medians = np.array([
        np.median(padded[i:i + window])
        for i in range(len(alt))
    ])

    return np.abs(alt - medians) <= threshold


# ============================================================
# 4. INITIAL REBOOST DETECTION
# ============================================================

def detect_reboosts(alt, window=3, threshold=0.5):
    """
    Detect sustained upward changes by comparing the medians
    of adjacent windows.

    Returns indices of candidate altitude jumps.
    """

    n = len(alt)
    if n < 2 * window:
        return np.array([], dtype=int)

    gains = {}

    for i in range(window, n - window + 1):
        previous = np.median(alt[i - window:i])
        following = np.median(alt[i:i + window])

        gain = following - previous

        if gain > threshold:
            gains[i] = gain

    jumps = []
    run = []

    for i in sorted(gains) + [None]:
        if run and (i is None or i != run[-1] + 1):
            k = max(run, key=lambda index: gains[index])

            candidates = range(
                max(1, k - window + 1),
                min(n, k + window),
            )

            if len(candidates):
                jump = max(
                    candidates,
                    key=lambda j: alt[j] - alt[j - 1],
                )
                jumps.append(jump)

            run = []

        if i is not None:
            run.append(i)

    return np.array(sorted(set(jumps)), dtype=int)


# ============================================================
# 5. MERGE CLOSE REBOOST CANDIDATES
# ============================================================

def merge_close_jumps(jumps, t_days, alt, min_gap_days=7):
    """
    Merge nearby candidate jumps.

    If candidates are less than min_gap_days apart, retain the
    candidate with the larger single-step altitude increase.

    Note: this is a filtering heuristic, not proof of a real burn.
    """

    if min_gap_days < 0:
        raise ValueError("min_gap_days cannot be negative.")

    jumps = sorted(set(int(j) for j in jumps))
    kept = []

    for j in jumps:
        if j <= 0 or j >= len(alt):
            continue

        if (
            kept
            and t_days[j] - t_days[kept[-1]] < min_gap_days
        ):
            previous_j = kept[-1]

            current_step = alt[j] - alt[j - 1]
            previous_step = alt[previous_j] - alt[previous_j - 1]

            if current_step > previous_step:
                kept[-1] = j
        else:
            kept.append(j)

    return np.array(kept, dtype=int)


# ============================================================
# 6. REBOOST VALIDATION / DIAGNOSTICS
# ============================================================

def validate_reboosts(jumps, epochs, t_days, alt, window=3):
    """
    Calculate diagnostic measurements for each retained candidate.

    These checks describe the shape of the altitude series.
    They do NOT independently confirm an actual spacecraft burn.
    """

    rows = []
    previous_day = None

    for j in jumps:
        # Single-step altitude change at the candidate
        step_gain = float(alt[j] - alt[j - 1])

        # Median altitude before and after the candidate
        pre_start = max(0, j - window)
        post_end = min(len(alt), j + window)

        before = alt[pre_start:j]
        after = alt[j:post_end]

        if len(before) and len(after):
            median_gain = float(
                np.median(after) - np.median(before)
            )
        else:
            median_gain = float("nan")

        if previous_day is None:
            gap_days = float("nan")
        else:
            gap_days = float(t_days[j] - previous_day)

        # Diagnostic only: does the local median also rise?
        if np.isfinite(median_gain) and median_gain > 0:
            diagnostic = "LOCAL_MEDIAN_RISE"
        else:
            diagnostic = "CHECK_MANUALLY"

        row = {
            "epoch": epochs[j].isoformat(),
            "index": int(j),
            "altitude_km": round(float(alt[j]), 4),
            "single_step_gain_km": round(step_gain, 4),
            "local_median_gain_km": (
                round(median_gain, 4)
                if np.isfinite(median_gain)
                else ""
            ),
            "days_since_previous_candidate": (
                round(gap_days, 2)
                if np.isfinite(gap_days)
                else ""
            ),
            "diagnostic": diagnostic,
        }

        rows.append(row)
        previous_day = t_days[j]

    return rows


def save_csv(path, fieldnames, rows):
    """Write a CSV file and verify that it exists and is nonempty."""

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not path.exists() or path.stat().st_size == 0:
        raise OSError(f"File was not created correctly: {path}")

    print(f"[OK] Saved: {path.resolve()}")
    print(f"     Size: {path.stat().st_size:,} bytes")


# ============================================================
# 7. MAIN ANALYSIS
# ============================================================

def main():
    print("=" * 65)
    print("ISS ALTITUDE ANALYSIS")
    print("=" * 65)
    print(f"[INFO] Output folder: {OUTPUT_DIR.resolve()}")

    # Download / load
    try:
        data = load_data()
    except (
        requests.RequestException,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"[ERROR] Could not load TLE history: {exc}")
        return

    if not data:
        print("[ERROR] No records returned.")
        return

    try:
        data = sorted(data, key=lambda item: item["EPOCH"])

        epochs_raw = np.array([
            datetime.fromisoformat(item["EPOCH"])
            for item in data
        ])

        mean_motion = np.array([
            float(item["MEAN_MOTION"])
            for item in data
        ])

    except (KeyError, TypeError, ValueError) as exc:
        print(f"[ERROR] Invalid history data: {exc}")
        return

    if len(epochs_raw) < 2:
        print("[ERROR] Not enough records.")
        return

    if not np.all(np.isfinite(mean_motion)) or np.any(mean_motion <= 0):
        print("[ERROR] Invalid mean-motion values in the history.")
        return

    t_days_raw = np.array([
        (epoch - epochs_raw[0]).total_seconds() / 86400
        for epoch in epochs_raw
    ])

    # Mean motion [rev/day] -> [rad/s] -> semi-major axis -> altitude
    n = mean_motion * 2 * np.pi / 86400
    semi_major_axis = (MU / n**2) ** (1.0 / 3.0)
    alt_raw = semi_major_axis - RE

    if not (350 < np.median(alt_raw) < 450):
        print(
            f"[WARN] Median altitude is {np.median(alt_raw):.1f} km. "
            "Check the satellite ID and units."
        )

    # --------------------------------------------------------
    # Outlier removal
    # --------------------------------------------------------

    if REMOVE_OUTLIERS:
        good = outlier_mask(
            alt_raw,
            window=OUTLIER_WINDOW,
            threshold=OUTLIER_KM,
        )
    else:
        good = np.ones(len(alt_raw), dtype=bool)

    bad = ~good
    bad_idx = np.where(bad)[0]

    print(f"\nOutliers flagged: {len(bad_idx)} / {len(alt_raw)}")

    removed_rows = [
        {
            "epoch": epochs_raw[i].isoformat(),
            "altitude_km": round(float(alt_raw[i]), 4),
        }
        for i in bad_idx
    ]

    try:
        save_csv(
            REMOVED_CSV,
            ["epoch", "altitude_km"],
            removed_rows,
        )
    except OSError as exc:
        print(f"[ERROR] Could not save outlier CSV: {exc}")

    # Export all original records, including flagged outliers
    altitude_rows = [
        {
            "epoch": epochs_raw[i].isoformat(),
            "altitude_km": round(float(alt_raw[i]), 4),
            "outlier_flagged": bool(bad[i]),
            "included_in_analysis": bool(good[i]),
        }
        for i in range(len(alt_raw))
    ]

    try:
        save_csv(
            ALTITUDE_CSV,
            [
                "epoch",
                "altitude_km",
                "outlier_flagged",
                "included_in_analysis",
            ],
            altitude_rows,
        )
    except OSError as exc:
        print(f"[ERROR] Could not save altitude CSV: {exc}")

    # Keep raw arrays for plotting; analyze only retained points
    epochs = epochs_raw[good]
    t_days = t_days_raw[good]
    alt = alt_raw[good]

    if len(alt) < max(2 * WINDOW, 5):
        print("[ERROR] Too few valid points after outlier removal.")
        return

    print(
        f"\nRecords analyzed: {len(alt)} / {len(alt_raw)}"
    )
    print(
        f"Period: {epochs[0]:%Y-%m-%d} -> {epochs[-1]:%Y-%m-%d}"
    )
    print(
        f"Altitude range: {alt.min():.2f} - {alt.max():.2f} km"
    )

    # Optional histogram
    if SHOW_HIST:
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(np.diff(alt), bins=200)
        ax_hist.set_yscale("log")
        ax_hist.set_xlabel(
            "Altitude change between consecutive TLEs [km]"
        )
        ax_hist.set_ylabel("Count")
        fig_hist.tight_layout()
        plt.show()
        plt.close(fig_hist)

    # --------------------------------------------------------
    # Detect candidates, then merge close candidates
    # --------------------------------------------------------

    raw_jumps = detect_reboosts(
        alt,
        window=WINDOW,
        threshold=REBOOST_THRESHOLD_KM,
    )

    jumps = merge_close_jumps(
        raw_jumps,
        t_days,
        alt,
        min_gap_days=MIN_GAP_DAYS,
    )

    print(f"\nInitial candidates: {len(raw_jumps)}")
    print(f"After {MIN_GAP_DAYS}-day merging: {len(jumps)}")

    print("\nRetained reboost candidates:")
    if len(jumps) == 0:
        print("  No candidates detected.")
    else:
        for j in jumps:
            step_gain = alt[j] - alt[j - 1]
            print(
                f"  {epochs[j]:%Y-%m-%d} | "
                f"altitude={alt[j]:.2f} km | "
                f"step change={step_gain:+.2f} km"
            )

    # --------------------------------------------------------
    # Validation diagnostics
    # --------------------------------------------------------

    validation_rows = validate_reboosts(
        jumps,
        epochs,
        t_days,
        alt,
        window=VALIDATION_WINDOW,
    )

    print("\nReboost validation diagnostics:")
    if not validation_rows:
        print("  No candidates to validate.")
    else:
        for row in validation_rows:
            print(
                f"  {row['epoch'][:10]} | "
                f"step={row['single_step_gain_km']:+.3f} km | "
                f"median gain={row['local_median_gain_km']} km | "
                f"gap={row['days_since_previous_candidate']} days | "
                f"{row['diagnostic']}"
            )

    try:
        save_csv(
            VALIDATION_CSV,
            [
                "epoch",
                "index",
                "altitude_km",
                "single_step_gain_km",
                "local_median_gain_km",
                "days_since_previous_candidate",
                "diagnostic",
            ],
            validation_rows,
        )
    except OSError as exc:
        print(f"[ERROR] Could not save validation CSV: {exc}")

    # --------------------------------------------------------
    # Fit decay rates between retained candidates
    # --------------------------------------------------------

    bounds = [0, *jumps.tolist(), len(alt)]
    rates = []
    fits = []

    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]

        if end - start < 5:
            continue

        if t_days[end - 1] - t_days[start] < 5:
            continue

        slope, intercept = np.polyfit(
            t_days[start:end],
            alt[start:end],
            1,
        )

        rates.append(slope)
        fits.append((start, end, slope, intercept))

    if rates:
        print(
            "\nMedian decay rate between candidates: "
            f"{np.median(rates) * 1000:.1f} m/day "
            f"({np.median(rates) * 30:.2f} km/month)"
        )
    else:
        print("\nNot enough segments to estimate decay rates.")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        epochs,
        alt,
        ".",
        markersize=2,
        label="Mean altitude (from TLE)",
    )

    for start, end, slope, intercept in fits:
        ax.plot(
            epochs[start:end],
            slope * t_days[start:end] + intercept,
            "r--",
            linewidth=1,
        )

    if len(jumps):
        ax.plot(
            epochs[jumps],
            alt[jumps],
            "g^",
            markersize=8,
            label="Merged reboost candidates",
        )

    if bad.any():
        ax.plot(
            epochs_raw[bad],
            alt_raw[bad],
            "rx",
            markersize=5,
            label="Flagged outliers",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Mean altitude [km]")
    ax.set_title("ISS altitude: atmospheric decay and reboost candidates")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    try:
        fig.savefig(
            PLOT_FILE,
            dpi=180,
            bbox_inches="tight",
        )

        if PLOT_FILE.exists() and PLOT_FILE.stat().st_size > 0:
            print(f"\n[OK] Saved PNG: {PLOT_FILE.resolve()}")
            print(f"     Size: {PLOT_FILE.stat().st_size:,} bytes")
        else:
            print("[ERROR] PNG was not created correctly.")

    except (OSError, ValueError) as exc:
        print(f"[ERROR] Could not save PNG: {exc}")

    finally:
        plt.show()
        plt.close(fig)

    print("\n" + "=" * 65)
    print("ANALYSIS FINISHED")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Plot: {PLOT_FILE.name}")
    print(f"All altitude data: {ALTITUDE_CSV.name}")
    print(f"Removed outliers: {REMOVED_CSV.name}")
    print(f"Reboost diagnostics: {VALIDATION_CSV.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
