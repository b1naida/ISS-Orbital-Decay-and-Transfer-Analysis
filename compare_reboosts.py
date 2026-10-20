"""Compare detected reboost candidates with a hand-made reference list.

Inputs (all next to this script):
  reboost_reference.csv   hand-made list: date_utc, vehicle, delta_v_m_s, source_url, verified
  reboost_validation.csv  written by the altitude script (epoch, single_step_gain_km, ...)
  iss_altitude_data.csv   (or iss_history.json) only used to get the time span of the data

Outputs: reboost_comparison.csv and reboost_comparison.md (a table for the README).
"""
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
REFERENCE_CSV = BASE / "reboost_reference.csv"
DETECTED_CSV = BASE / "reboost_validation.csv"
ALTITUDE_CSV = BASE / "iss_altitude_data.csv"
CACHE = BASE / "iss_history.json"
OUT_CSV = BASE / "reboost_comparison.csv"
OUT_MD = BASE / "reboost_comparison.md"

TOLERANCE_DAYS = 3.0   # detected epoch may lag the real burn by a day or two
KM_PER_MS = 1.8        # rough altitude gain per 1 m/s of delta-v at ISS altitude

COLUMNS = ["status", "reference_date", "detected_epoch", "lag_days", "jump_km",
           "expected_km", "vehicle", "delta_v_m_s", "verified"]


def parse_dt(text):
    return datetime.fromisoformat(text.strip().replace("Z", ""))


def read_reference():
    rows = []
    with REFERENCE_CSV.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d = (r.get("date_utc") or "").strip()
            if not d or d.startswith("#"):
                continue
            dv = (r.get("delta_v_m_s") or "").strip()
            rows.append({
                "dt": parse_dt(d),
                "vehicle": (r.get("vehicle") or "").strip(),
                "dv": float(dv) if dv else None,
                "verified": (r.get("verified") or "").strip(),
            })
    return rows


def read_detected():
    rows = []
    with DETECTED_CSV.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "dt": parse_dt(r["epoch"]),
                "jump": float(r["single_step_gain_km"]),
            })
    return rows


def data_span():
    """(first, last) epoch of the analysed data, or None if unknown."""
    try:
        if ALTITUDE_CSV.exists():
            with ALTITUDE_CSV.open(newline="", encoding="utf-8-sig") as f:
                epochs = [parse_dt(r["epoch"]) for r in csv.DictReader(f)]
        elif CACHE.exists():
            epochs = [parse_dt(d["EPOCH"]) for d in json.loads(CACHE.read_text(encoding="utf-8"))]
        else:
            return None
        return min(epochs), max(epochs)
    except (KeyError, ValueError, OSError):
        return None


def match(ref, det):
    """One-to-one matching, closest pairs first, within TOLERANCE_DAYS."""
    pairs = sorted(
        (abs((d["dt"] - r["dt"]).total_seconds()) / 86400, i, j)
        for i, r in enumerate(ref) for j, d in enumerate(det)
    )
    ref_to_det, det_to_ref = {}, {}
    for diff, i, j in pairs:
        if diff > TOLERANCE_DAYS:
            break
        if i in ref_to_det or j in det_to_ref:
            continue
        ref_to_det[i] = j
        det_to_ref[j] = i
    return ref_to_det, det_to_ref


def main():
    for path in (REFERENCE_CSV, DETECTED_CSV):
        if not path.exists():
            print(f"[ERROR] Missing file: {path.name}. Run the altitude script / fill the reference list first.")
            return

    try:
        ref, det = read_reference(), read_detected()
    except (KeyError, ValueError, OSError) as exc:
        print(f"[ERROR] Could not read the CSV files: {exc}")
        return
    if not ref:
        print("[ERROR] reboost_reference.csv has no rows.")
        return

    span = data_span()
    if span is None:
        print("[WARN] Data time span unknown; reference rows outside the data cannot be excluded.")
    in_range = [span is None or span[0] <= r["dt"] <= span[1] for r in ref]

    ref_to_det, det_to_ref = match(ref, det)

    rows = []
    for i, r in enumerate(ref):
        expected = round(KM_PER_MS * r["dv"], 2) if r["dv"] is not None else ""
        base = {"reference_date": r["dt"].strftime("%Y-%m-%d"), "expected_km": expected,
                "vehicle": r["vehicle"], "delta_v_m_s": "" if r["dv"] is None else r["dv"],
                "verified": r["verified"]}
        if not in_range[i]:
            rows.append({**base, "status": "OUT_OF_RANGE", "detected_epoch": "", "lag_days": "", "jump_km": ""})
        elif i in ref_to_det:
            d = det[ref_to_det[i]]
            lag = (d["dt"] - r["dt"]).total_seconds() / 86400
            rows.append({**base, "status": "MATCH", "detected_epoch": d["dt"].strftime("%Y-%m-%d %H:%M"),
                         "lag_days": round(lag, 2), "jump_km": round(d["jump"], 2)})
        else:
            rows.append({**base, "status": "MISSED", "detected_epoch": "", "lag_days": "", "jump_km": ""})
    for j, d in enumerate(det):
        if j not in det_to_ref:
            rows.append({"status": "EXTRA", "reference_date": "", "detected_epoch": d["dt"].strftime("%Y-%m-%d %H:%M"),
                         "lag_days": "", "jump_km": round(d["jump"], 2), "expected_km": "",
                         "vehicle": "", "delta_v_m_s": "", "verified": ""})

    rows.sort(key=lambda r: r["reference_date"] or r["detected_epoch"][:10])

    # --- Print and save ---
    widths = [12, 11, 17, 8, 8, 11, 14, 8, 8]
    print(" ".join(c.ljust(w) for c, w in zip(COLUMNS, widths)))
    for r in rows:
        print(" ".join(str(r[c]).ljust(w) for c, w in zip(COLUMNS, widths)))

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("|" + "---|" * len(COLUMNS) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in COLUMNS) + " |\n")

    # --- Summary ---
    n_ref = sum(in_range)
    n_match = len(ref_to_det) if span is None else sum(1 for i in ref_to_det if in_range[i])
    lags = sorted(r["lag_days"] for r in rows if r["status"] == "MATCH")
    print(f"\nReference rows in data range: {n_ref}  (outside range: {len(ref) - n_ref})")
    print(f"Matched within +/-{TOLERANCE_DAYS:g} days: {n_match}"
          + (f"  -> recall {n_match / n_ref:.0%}" if n_ref else ""))
    print(f"Detected candidates: {len(det)}  (confirmed by the reference list: {len(det_to_ref)}, "
          f"not in the list: {len(det) - len(det_to_ref)})")
    if lags:
        print(f"Detected epoch minus reference date: median {statistic.median(lags)]:+.2f} days "
              f"(min {lags[0]:+.2f}, max {lags[-1]:+.2f})")
    print("Note: the reference list is hand-made and incomplete, so 'not in the list' does NOT mean false positive.")
    print(f"\n[OK] Saved: {OUT_CSV.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
