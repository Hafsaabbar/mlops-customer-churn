#!/usr/bin/env python3
"""
Traffic generator for Customer Churn Prediction API load scenarios.

Modes:
  stable  - Send every row of X_test.csv exactly once; all must return 200.
  invalid - Mix valid rows with intentionally corrupt payloads (expect 422).
  drift   - Send contract-valid but distribution-shifted rows; none must return 422.

Only Python standard library is used.
"""

import argparse
import csv
import collections
import json
import random
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path):
    """Return (header list, list of row dicts) from a CSV file."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows


def _coerce_row(row):
    """Cast numeric fields to their expected Python types."""
    out = dict(row)
    out["SeniorCitizen"] = int(out["SeniorCitizen"])
    out["tenure"] = int(out["tenure"])
    out["MonthlyCharges"] = float(out["MonthlyCharges"])
    out["TotalCharges"] = float(out["TotalCharges"])
    # customerID is optional - keep as string
    return out


def _send(base_url, payload):
    """POST payload to /predict. Returns (status_code, latency_seconds)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.read()
    except Exception:
        status = 0
    latency = time.monotonic() - t0
    return status, latency


def _percentile(values, pct):
    """Compute a percentile (0-100) over a list of floats."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * pct / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


def _print_summary(mode, sent, code_counts, latencies, duration, rate_limit):
    real_rate = sent / duration if duration > 0 else 0.0
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    print("")
    print("=" * 60)
    print("TRAFFIC GENERATOR SUMMARY")
    print("=" * 60)
    print(f"  mode       : {mode}")
    print(f"  sent       : {sent}")
    print(f"  duration   : {duration:.1f}s")
    print(f"  target rate: {rate_limit} req/s")
    print(f"  actual rate: {real_rate:.1f} req/s")
    print(f"  HTTP codes : {dict(sorted(code_counts.items()))}")
    print(f"  p50 latency: {p50 * 1000:.1f} ms")
    print(f"  p95 latency: {p95 * 1000:.1f} ms")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Corrupt helpers (for invalid mode)
# ---------------------------------------------------------------------------

_CORRUPT_CYCLE = 3  # three alternating corruption kinds

def _corrupt(row, idx):
    """Return a corrupted copy of a row (guaranteed to produce 422)."""
    r = dict(row)
    kind = idx % _CORRUPT_CYCLE
    if kind == 0:
        r["Contract"] = "Lifetime"           # invalid enum value
    elif kind == 1:
        r["tenure"] = -1                     # below ge=0
    else:
        r.pop("MonthlyCharges", None)        # missing required field
    return r


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def run_stable(base_url, rows, rate, duration, rng):
    """Send every row exactly once; all must return 200."""
    sent = 0
    code_counts = collections.Counter()
    latencies = []
    interval = 1.0 / rate if rate > 0 else 0.0
    t_start = time.monotonic()

    for row in rows:
        payload = _coerce_row(row)
        status, lat = _send(base_url, payload)
        sent += 1
        code_counts[status] += 1
        latencies.append(lat)

        if duration is not None and (time.monotonic() - t_start) >= duration:
            break

        elapsed = time.monotonic() - t_start
        expected = sent * interval
        if expected > elapsed:
            time.sleep(expected - elapsed)

    actual_duration = time.monotonic() - t_start
    _print_summary("stable", sent, code_counts, latencies, actual_duration, rate)

    non_200 = sum(v for k, v in code_counts.items() if k != 200)
    if non_200 > 0:
        print(f"[ERROR] {non_200} non-200 responses in stable mode.")
        return 1
    return 0


def run_invalid(base_url, rows, rate, duration, invalid_ratio, rng):
    """Send rows, corrupting a fraction to produce 422s."""
    sent = 0
    code_counts = collections.Counter()
    latencies = []
    interval = 1.0 / rate if rate > 0 else 0.0
    t_start = time.monotonic()
    idx = 0
    row_pool = rows.copy()
    rng.shuffle(row_pool)
    corrupt_idx = 0

    while True:
        if duration is not None and (time.monotonic() - t_start) >= duration:
            break
        row = row_pool[idx % len(row_pool)]
        idx += 1
        should_corrupt = rng.random() < invalid_ratio
        payload = _coerce_row(row) if not should_corrupt else _corrupt(_coerce_row(row), corrupt_idx)
        if should_corrupt:
            corrupt_idx += 1

        status, lat = _send(base_url, payload)
        sent += 1
        code_counts[status] += 1
        latencies.append(lat)

        elapsed = time.monotonic() - t_start
        expected = sent * interval
        if expected > elapsed:
            time.sleep(expected - elapsed)

    actual_duration = time.monotonic() - t_start
    _print_summary("invalid", sent, code_counts, latencies, actual_duration, rate)

    if 422 not in code_counts:
        print("[ERROR] No 422 responses received in invalid mode (expected some).")
        return 1
    return 0


def run_drift(base_url, rows, rate, duration, rng):
    """Send distribution-shifted but schema-valid rows; none must return 422."""
    sent = 0
    code_counts = collections.Counter()
    latencies = []
    interval = 1.0 / rate if rate > 0 else 0.0
    t_start = time.monotonic()
    idx = 0
    row_pool = rows.copy()

    while True:
        if duration is not None and (time.monotonic() - t_start) >= duration:
            break

        row = _coerce_row(row_pool[idx % len(row_pool)])
        idx += 1

        # Apply distribution shift while keeping contract validity
        # 90% forced to Month-to-month
        if rng.random() < 0.9:
            row["Contract"] = "Month-to-month"

        # Divide tenure by 4 (rounded), minimum 0
        new_tenure = max(0, round(row["tenure"] / 4))
        row["tenure"] = new_tenure

        # Inflate monthly charges by 1.4, capped at 150.0
        new_monthly = min(150.0, row["MonthlyCharges"] * 1.4)
        row["MonthlyCharges"] = round(new_monthly, 2)

        # Recompute TotalCharges: tenure * monthly, capped at 10000.0
        if new_tenure == 0:
            row["TotalCharges"] = 0.0
        else:
            row["TotalCharges"] = min(10000.0, round(new_tenure * new_monthly, 2))

        # Cross-rule: if InternetService == "No", ensure add-ons are "No internet service"
        # (original rows are already valid, but Contract/tenure changes don't affect this)
        # Only Phone cross-rule needs care: if PhoneService == "No" ensure MultipleLines correct
        if row["PhoneService"] == "No":
            row["MultipleLines"] = "No phone service"

        status, lat = _send(base_url, payload=row)
        sent += 1
        code_counts[status] += 1
        latencies.append(lat)

        elapsed = time.monotonic() - t_start
        expected = sent * interval
        if expected > elapsed:
            time.sleep(expected - elapsed)

    actual_duration = time.monotonic() - t_start
    _print_summary("drift", sent, code_counts, latencies, actual_duration, rate)

    if 422 in code_counts and code_counts[422] > 0:
        print(f"[ERROR] {code_counts[422]} unexpected 422 response(s) in drift mode.")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Traffic generator for Churn Prediction API load scenarios."
    )
    parser.add_argument(
        "mode",
        choices=["stable", "invalid", "drift"],
        help="Traffic scenario to run.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the API service (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--data",
        default="data/splits/test/X_test.csv",
        help="Path to X_test.csv (default: data/splits/test/X_test.csv).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="Target request rate in requests/second (default: 20).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Maximum duration in seconds. Omit to send all rows exactly once (stable) or run indefinitely.",
    )
    parser.add_argument(
        "--invalid-ratio",
        type=float,
        default=0.4,
        dest="invalid_ratio",
        help="Fraction of requests to corrupt in invalid mode (default: 0.4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    rows = _load_csv(args.data)
    if not rows:
        print("[ERROR] CSV file is empty or not found.")
        sys.exit(1)

    if args.mode == "stable":
        exit_code = run_stable(args.base_url, rows, args.rate, args.duration, rng)
    elif args.mode == "invalid":
        exit_code = run_invalid(
            args.base_url, rows, args.rate, args.duration, args.invalid_ratio, rng
        )
    else:
        exit_code = run_drift(args.base_url, rows, args.rate, args.duration, rng)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
