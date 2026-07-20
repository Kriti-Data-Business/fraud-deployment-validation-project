"""Feature engineering for application-fraud detection.

All features are computed *point-in-time*: for each application they use only
applications that arrived earlier in the stream. This avoids target/temporal
leakage — the exact property the Part B validation will later test for.

Feature families:
- **Velocity**: applications sharing this device/address/phone/email in the
  trailing 7 and 30 days (fraud rings burst; legitimate entities rarely do).
- **Entity degree (expanding)**: distinct emails seen on this device so far,
  distinct devices on this address, etc. (synthetic-identity linkage signal).
- **Affordability / plausibility**: requested amount vs stated income,
  employment history vs age.
- **Behavioural**: submission hour, night-time flag, channel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENTITY_COLS = ["device_id", "address_id", "phone", "email"]
CATEGORICAL = ["channel"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must be sorted by ts (synth_data.generate guarantees this)."""
    out = df.copy()
    ts_num = out["ts"].astype("int64").to_numpy() // 10**9

    for col in ENTITY_COLS:
        groups = out.groupby(col, sort=False).indices
        for days in (7, 30):
            out[f"{col}_apps_{days}d"] = _trailing_counts(groups, ts_num, days)

    out["emails_per_device"] = _expanding_distinct(out, "device_id", "email")
    out["devices_per_address"] = _expanding_distinct(out, "address_id", "device_id")
    out["phones_per_device"] = _expanding_distinct(out, "device_id", "phone")

    out["amount_income_ratio"] = out["requested_amount"] / (out["income"] + 1.0)
    out["employment_age_ratio"] = out["employment_years"] / (out["age"] - 17.0)
    out["is_free_email"] = out["email"].str.split("@").str[1].isin(
        ["gmail.com", "outlook.com", "yahoo.com", "proton.me"]
    ).astype(int)

    out["hour"] = out["ts"].dt.hour
    out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
    out["existing_customer"] = out["existing_customer"].astype(int)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "age", "income", "employment_years", "requested_amount",
        "existing_customer", "amount_income_ratio", "employment_age_ratio",
        "is_free_email", "hour", "is_night",
        "emails_per_device", "devices_per_address", "phones_per_device",
    ]
    velocity = [f"{c}_apps_{d}d" for c in ENTITY_COLS for d in (7, 30)]
    return base + velocity + CATEGORICAL


def _trailing_counts(groups: dict, ts_num: np.ndarray, days: int) -> np.ndarray:
    """Count of *prior* applications for the same entity within the window."""
    out = np.zeros(len(ts_num), dtype=np.int32)
    window = days * 86_400
    for idx in groups.values():
        t = ts_num[idx]  # increasing, because the frame is sorted by ts
        left = np.searchsorted(t, t - window, side="left")
        out[idx] = np.arange(len(t)) - left
    return out


def _expanding_distinct(df: pd.DataFrame, key: str, value: str) -> np.ndarray:
    """Distinct `value`s seen for this `key` up to and including each row."""
    out = np.zeros(len(df), dtype=np.int32)
    seen: dict[str, set] = {}
    keys = df[key].to_numpy()
    values = df[value].to_numpy()
    for i in range(len(df)):
        s = seen.setdefault(keys[i], set())
        s.add(values[i])
        out[i] = len(s)
    return out
