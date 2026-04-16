"""Preprocess smart-meter CSVs into clean hourly load series.

Supports two input schemas automatically:
1) Interval schema: x_Timestamp, t_kWh, meter (e.g., 3-minute records)
2) Daily schema: Date, t_kWh, meter (daily totals)
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_GLOB = str(PROJECT_ROOT / "Dataset" / "*.csv")
DEFAULT_OUT_PATH = PROJECT_ROOT / "outputs" / "cleaned_hourly.csv"

# Typical residential hourly load share. Sum is 1.0.
DEFAULT_HOURLY_PROFILE = [
    0.018,
    0.016,
    0.015,
    0.015,
    0.016,
    0.019,
    0.028,
    0.040,
    0.050,
    0.053,
    0.049,
    0.044,
    0.039,
    0.036,
    0.035,
    0.036,
    0.040,
    0.050,
    0.063,
    0.073,
    0.080,
    0.076,
    0.059,
    0.050,
]


def _aggregate_interval_files(
    csv_paths: list[str],
    combine_mode: str,
    timestamp_col: str,
    energy_col: str,
    chunksize: int,
) -> pd.Series:
    """Aggregate interval data to a timestamp-level series across all files."""
    total_sum = pd.Series(dtype="float64")
    total_count = pd.Series(dtype="float64")

    for path in csv_paths:
        usecols = [timestamp_col, energy_col]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            ts = pd.to_datetime(chunk[timestamp_col], errors="coerce")
            val = pd.to_numeric(chunk[energy_col], errors="coerce")
            valid = ts.notna() & val.notna()
            if not valid.any():
                continue

            slim = pd.DataFrame({timestamp_col: ts[valid], energy_col: val[valid]})
            grouped_sum = slim.groupby(timestamp_col)[energy_col].sum()

            total_sum = total_sum.add(grouped_sum, fill_value=0.0)
            if combine_mode == "mean":
                grouped_cnt = slim.groupby(timestamp_col)[energy_col].count().astype(float)
                total_count = total_count.add(grouped_cnt, fill_value=0.0)

    if len(total_sum) == 0:
        raise ValueError("No valid interval records found in input files")

    total_sum = total_sum.sort_index()
    if combine_mode == "sum":
        return total_sum

    total_count = total_count.reindex(total_sum.index, fill_value=1.0)
    return total_sum / total_count


def _expand_daily_files(
    df: pd.DataFrame,
    date_col: str,
    meter_col: str,
    energy_col: str,
    combine_mode: str,
) -> pd.Series:
    """Expand daily totals into hourly values using a fixed load profile."""
    profile = pd.Series(DEFAULT_HOURLY_PROFILE, dtype=float)
    profile = profile / profile.sum()

    expanded = []
    for hour in range(24):
        part = df[[meter_col, date_col, energy_col]].copy()
        part[date_col] = part[date_col] + pd.to_timedelta(hour, unit="h")
        part[energy_col] = part[energy_col] * float(profile.iloc[hour])
        expanded.append(part)
    hourly = pd.concat(expanded, ignore_index=True)

    agg_fn = "mean" if combine_mode == "mean" else "sum"
    series = hourly.groupby(date_col, as_index=True)[energy_col].agg(agg_fn).sort_index()
    return series


def preprocess_hourly(
    input_glob: str,
    out_path: Path,
    date_col: str = "Date",
    meter_col: str = "meter",
    energy_col: str = "t_kWh",
    combine_mode: str = "mean",
    fill_missing: bool = True,
    chunksize: int = 300000,
) -> pd.DataFrame:
    """Load meter CSVs and return a merged hourly energy dataframe."""
    csv_paths = sorted(glob(input_glob))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched: {input_glob}")

    if combine_mode not in {"mean", "sum"}:
        raise ValueError("combine_mode must be either 'mean' or 'sum'")

    sample_cols = set(pd.read_csv(csv_paths[0], nrows=1).columns)

    if "x_Timestamp" in sample_cols:
        timestamp_col = "x_Timestamp"
        interval_series = _aggregate_interval_files(
            csv_paths=csv_paths,
            combine_mode=combine_mode,
            timestamp_col=timestamp_col,
            energy_col=energy_col,
            chunksize=chunksize,
        )
        hourly = interval_series.resample("h").sum(min_count=1)
    elif date_col in sample_cols:
        frames = []
        required = {meter_col, date_col, energy_col}
        for path in csv_paths:
            frame = pd.read_csv(path)
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{Path(path).name} missing required columns: {sorted(missing)}")
            frames.append(frame)

        df = pd.concat(frames, ignore_index=True)
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")
        df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
        df = df.dropna(subset=[meter_col, date_col, energy_col])
        df = df.sort_values([meter_col, date_col])
        df = df.drop_duplicates(subset=[meter_col, date_col], keep="last")
        hourly = _expand_daily_files(df, date_col, meter_col, energy_col, combine_mode)
        hourly = hourly.resample("h").sum(min_count=1)
    else:
        raise ValueError(
            "Unsupported input schema. Expected either 'x_Timestamp' or 'Date' with t_kWh"
        )

    if fill_missing:
        hourly = hourly.interpolate(method="time").ffill().bfill()

    cleaned = hourly.to_frame(name=energy_col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index_label="x_Timestamp")
    return cleaned


def main() -> None:
    """CLI entrypoint for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess daily smart meter CSVs to hourly kWh.")
    parser.add_argument("--input-glob", type=str, default=DEFAULT_INPUT_GLOB, help="Glob for input CSV files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_PATH, help="Hourly CSV output path")
    parser.add_argument(
        "--combine-mode",
        type=str,
        choices=["mean", "sum"],
        default="mean",
        help="How to combine multiple meter readings per hour",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=300000,
        help="Chunk size for interval-schema CSV processing",
    )
    parser.add_argument(
        "--no-fill-missing",
        action="store_true",
        help="Disable interpolation/forward-backward fill for missing hourly values",
    )
    args = parser.parse_args()

    cleaned = preprocess_hourly(
        input_glob=args.input_glob,
        out_path=args.output,
        combine_mode=args.combine_mode,
        fill_missing=not args.no_fill_missing,
        chunksize=args.chunksize,
    )

    print("Hourly data saved")
    print(f"Rows: {len(cleaned)}")
    print(f"Total energy (kWh): {cleaned['t_kWh'].sum():.3f}")


if __name__ == "__main__":
    main()
