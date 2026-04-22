from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _pick_column(columns: Iterable[str], candidates: list[str]) -> str:
    index = {c.lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in index:
            return index[key]
    raise ValueError(f"Could not find any of columns: {candidates}")


def _resolve_input_csv(path: str) -> Path:
    """Resolve input CSV path from common project locations.

    Resolution order:
    1) Exact path as provided
    2) Current working directory + basename
    3) outputs/ + basename
    4) Project root outputs/ + basename (when called from subdirs)
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate

    basename = candidate.name
    attempts = [
        Path.cwd() / basename,
        Path.cwd() / "outputs" / basename,
    ]

    # If running inside physics_engine, this covers ../outputs/*.csv
    if Path(__file__).resolve().parent.name == "physics_engine":
        attempts.append(Path(__file__).resolve().parent.parent / "outputs" / basename)

    for attempt in attempts:
        if attempt.exists():
            return attempt

    searched = [str(candidate)] + [str(p) for p in attempts]
    raise FileNotFoundError(
        "CSV file not found. Tried: " + ", ".join(searched)
    )


def load_load_data(path: str) -> pd.DataFrame:
    resolved = _resolve_input_csv(path)
    df = pd.read_csv(resolved)
    ts_col = _pick_column(df.columns, ["x_Timestamp", "timestamp", "timestamp_utc", "datetime"])
    load_col = _pick_column(df.columns, ["t_kWh", "load_kwh", "consumption_kwh", "kwh", "load"])

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
            "load_kwh": pd.to_numeric(df[load_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp", "load_kwh"]).reset_index(drop=True)
    return out


def load_weather_data(path: str) -> pd.DataFrame:
    resolved = _resolve_input_csv(path)
    df = pd.read_csv(resolved)
    ts_col = _pick_column(df.columns, ["timestamp_utc", "x_Timestamp", "timestamp", "datetime"])
    ghi_col = _pick_column(df.columns, ["ghi_kwh_m2", "ghi_w_m2", "ghi"])

    wind_col = None
    for c in ["wind_speed_10m_mps", "ws10m", "wind_mps", "wind_speed"]:
        if c in df.columns:
            wind_col = c
            break

    temp_col = None
    for c in ["temperature_2m_c", "temp_c", "t2m_c", "ambient_temp_c"]:
        if c in df.columns:
            temp_col = c
            break

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
            # NASA POWER values are W/m^2 even when legacy naming says kwh_m2.
            "ghi_w_m2": pd.to_numeric(df[ghi_col], errors="coerce"),
            "wind_mps": pd.to_numeric(df[wind_col], errors="coerce") if wind_col else 0.0,
            "ambient_temp_c": pd.to_numeric(df[temp_col], errors="coerce") if temp_col else float("nan"),
        }
    )
    out = out.dropna(subset=["timestamp", "ghi_w_m2"]).reset_index(drop=True)
    return out


def align_inputs(load_df: pd.DataFrame, weather_df: pd.DataFrame, mode: str = "auto") -> pd.DataFrame:
    mode = mode.lower().strip()

    if mode not in {"auto", "timestamp", "by_order"}:
        raise ValueError("align mode must be one of: auto, timestamp, by_order")

    if mode in {"auto", "timestamp"}:
        merged = pd.merge(load_df, weather_df, on="timestamp", how="inner")
        if mode == "timestamp":
            if merged.empty:
                raise ValueError("No overlapping timestamps between load and weather data")
            return merged.sort_values("timestamp").reset_index(drop=True)
        if not merged.empty:
            return merged.sort_values("timestamp").reset_index(drop=True)

    n = min(len(load_df), len(weather_df))
    if n == 0:
        raise ValueError("Input data is empty after cleaning")

    lhs = load_df.iloc[:n].reset_index(drop=True)
    rhs = weather_df.iloc[:n].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "timestamp": lhs["timestamp"],
            "load_kwh": lhs["load_kwh"],
            "ghi_w_m2": rhs["ghi_w_m2"],
            "wind_mps": rhs["wind_mps"],
            "ambient_temp_c": rhs["ambient_temp_c"],
        }
    )
    return out
