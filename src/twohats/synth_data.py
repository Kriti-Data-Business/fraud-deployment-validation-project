"""Synthetic application-fraud dataset generator.

Generates a realistic loan/credit application stream containing two fraud
patterns observed in application fraud:

- **Ring / synthetic-identity fraud**: bursts of applications from small
  clusters of shared devices, addresses and phones, with thin-file
  applicants and inflated incomes.
- **Opportunistic first-party fraud**: singleton applications with
  misrepresented income relative to requested amount.

Ground truth is known by construction, which is what makes the Part B
independent-validation exercise provable (we can check the validator's
findings against how the data was actually generated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FREE_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "proton.me"]
PAID_DOMAINS = ["bigpond.com.au", "company.com.au", "corp.au", "edu.au"]
CHANNELS = ["online", "branch", "broker", "mobile_app"]


def _emails(rng: np.random.Generator, n: int, free_ratio: float) -> np.ndarray:
    domains = np.where(
        rng.random(n) < free_ratio,
        rng.choice(FREE_DOMAINS, n),
        rng.choice(PAID_DOMAINS, n),
    )
    local = rng.integers(10_000, 99_999_999, n)
    return np.char.add(np.char.add(local.astype(str), "@"), domains.astype(str))


def generate(
    n_apps: int = 80_000,
    fraud_rate: float = 0.025,
    n_days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the application stream, sorted by timestamp."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n_apps * fraud_rate)
    n_ring = int(n_fraud * 0.6)
    n_opp = n_fraud - n_ring
    n_legit = n_apps - n_fraud

    frames = [
        _legit(rng, n_legit, n_days),
        _opportunistic(rng, n_opp, n_days),
        _rings(rng, n_ring, n_days),
    ]
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df.pop("ts_sec"), unit="s")
    df = df.sort_values("ts", ignore_index=True)
    df.insert(0, "app_id", np.arange(len(df)))
    return df


def _base_frame(n: int) -> dict:
    return {k: np.empty(n, dtype=object) for k in ()}


def _legit(rng, n, n_days):
    # Mostly-unique entities; some benign sharing (families share an address,
    # occasionally a device).
    age = rng.normal(42, 13, n).clip(18, 85)
    income = np.exp(rng.normal(11.2, 0.5, n)).clip(28_000, 600_000)
    amount = (income * rng.uniform(0.05, 0.6, n)).clip(1_000, 300_000)
    employment = (age - 18) * rng.beta(3, 3, n)
    hour = rng.choice(24, n, p=_day_hours())
    return pd.DataFrame(
        {
            "ts_sec": rng.integers(0, n_days, n) * 86_400 + hour * 3_600 + rng.integers(0, 3_600, n),
            "device_id": np.char.add("d", rng.integers(0, int(n * 0.92), n).astype(str)),
            "address_id": np.char.add("a", rng.integers(0, int(n * 0.55), n).astype(str)),
            "phone": np.char.add("p", rng.integers(0, int(n * 0.95), n).astype(str)),
            "email": _emails(rng, n, free_ratio=0.55),
            "channel": rng.choice(CHANNELS, n, p=[0.45, 0.2, 0.2, 0.15]),
            "age": age.round(0),
            "income": income.round(0),
            "employment_years": employment.round(1),
            "requested_amount": amount.round(-2),
            "existing_customer": rng.random(n) < 0.45,
            "is_fraud": np.zeros(n, dtype=int),
            "fraud_type": np.full(n, "none"),
        }
    )


def _opportunistic(rng, n, n_days):
    # Looks like a normal applicant, but income is overstated relative to
    # reality: high amount-to-(true)income, short employment history.
    age = rng.normal(36, 11, n).clip(18, 75)
    income = np.exp(rng.normal(11.6, 0.45, n)).clip(40_000, 700_000)  # stated (inflated)
    amount = (income * rng.uniform(0.35, 1.1, n)).clip(5_000, 400_000)
    employment = (age - 18) * rng.beta(1.3, 5, n)
    hour = rng.choice(24, n, p=_day_hours())
    return pd.DataFrame(
        {
            "ts_sec": rng.integers(0, n_days, n) * 86_400 + hour * 3_600 + rng.integers(0, 3_600, n),
            "device_id": np.char.add("d", rng.integers(0, int(n * 50), n).astype(str)),
            "address_id": np.char.add("a", rng.integers(0, int(n * 30), n).astype(str)),
            "phone": np.char.add("p", rng.integers(0, int(n * 50), n).astype(str)),
            "email": _emails(rng, n, free_ratio=0.75),
            "channel": rng.choice(CHANNELS, n, p=[0.6, 0.05, 0.2, 0.15]),
            "age": age.round(0),
            "income": income.round(0),
            "employment_years": employment.round(1),
            "requested_amount": amount.round(-2),
            "existing_customer": rng.random(n) < 0.15,
            "is_fraud": np.ones(n, dtype=int),
            "fraud_type": np.full(n, "opportunistic"),
        }
    )


def _rings(rng, n, n_days):
    # Fraud rings: 5-25 applications sharing a small pool of devices,
    # addresses and phones, submitted in a burst over ~2 weeks, skewed to
    # night hours, thin-file young applicants, inflated income.
    rows = []
    ring_id = 0
    remaining = n
    while remaining > 0:
        size = int(min(remaining, rng.integers(5, 26)))
        remaining -= size
        devices = [f"rd{ring_id}_{i}" for i in range(int(rng.integers(1, 5)))]
        addresses = [f"ra{ring_id}_{i}" for i in range(int(rng.integers(1, 4)))]
        phones = [f"rp{ring_id}_{i}" for i in range(int(rng.integers(1, 6)))]
        start = rng.integers(0, max(1, n_days - 14)) * 86_400
        age = rng.normal(27, 6, size).clip(18, 55)
        income = np.exp(rng.normal(11.7, 0.35, size)).clip(60_000, 500_000)
        hour = rng.choice(24, size, p=_night_hours())
        rows.append(
            pd.DataFrame(
                {
                    "ts_sec": start + rng.integers(0, 14, size) * 86_400 + hour * 3_600 + rng.integers(0, 3_600, size),
                    "device_id": rng.choice(devices, size),
                    "address_id": rng.choice(addresses, size),
                    "phone": rng.choice(phones, size),
                    "email": _emails(rng, size, free_ratio=0.95),
                    "channel": rng.choice(CHANNELS, size, p=[0.75, 0.02, 0.08, 0.15]),
                    "age": age.round(0),
                    "income": income.round(0),
                    "employment_years": ((age - 18) * rng.beta(1.2, 8, size)).round(1),
                    "requested_amount": (income * rng.uniform(0.3, 0.9, size)).clip(8_000, 350_000).round(-2),
                    "existing_customer": rng.random(size) < 0.05,
                    "is_fraud": np.ones(size, dtype=int),
                    "fraud_type": np.full(size, "ring"),
                }
            )
        )
        ring_id += 1
    return pd.concat(rows, ignore_index=True)


def _day_hours():
    p = np.ones(24)
    p[8:21] = 4.0
    return p / p.sum()


def _night_hours():
    p = np.ones(24)
    p[0:6] = 3.5
    p[21:24] = 3.0
    return p / p.sum()
