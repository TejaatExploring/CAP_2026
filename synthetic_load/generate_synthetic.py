"""Generate a 24x30 synthetic hourly load CSV from trained Brain 1a artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAILY_PATH = PROJECT_ROOT / "outputs" / "daily_profiles.npy"
DEFAULT_HOURLY_PATH = PROJECT_ROOT / "outputs" / "cleaned_hourly.csv"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "kmeans_model.pkl"
DEFAULT_MARKOV_PATH = Path(__file__).resolve().parent / "markov_transition.npy"
DEFAULT_META_PATH = Path(__file__).resolve().parent / "metadata.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "outputs" / "synthetic_30d_hourly.csv"


def cluster_hour_std(daily: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Estimate per-cluster hourly variability from historical daily profiles."""
    stds = np.zeros((n_clusters, 24), dtype=float)
    global_std = np.maximum(np.std(daily, axis=0), 1e-3)
    for c in range(n_clusters):
        members = daily[labels == c]
        if len(members) < 2:
            stds[c] = global_std
        else:
            stds[c] = np.maximum(np.std(members, axis=0), 1e-3)
    return stds


def generate_series(
    days: int,
    centers: np.ndarray,
    transition: np.ndarray,
    stds: np.ndarray,
    start_cluster: int,
    seed: int,
    noise_scale: float,
) -> np.ndarray:
    """Generate an hourly synthetic load series from cluster centers and transitions."""
    rng = np.random.default_rng(seed)
    cluster = int(start_cluster)
    values: list[float] = []

    for _ in range(days):
        noise = rng.normal(0.0, stds[cluster] * noise_scale)
        day_profile = np.clip(centers[cluster] + noise, 0.0, None)
        values.extend(day_profile.tolist())
        cluster = int(rng.choice(len(centers), p=transition[cluster]))

    return np.asarray(values, dtype=float)


def build_weekday_transitions(
    labels: np.ndarray,
    dates: np.ndarray,
    n_clusters: int,
    laplace: float = 1.0,
) -> np.ndarray:
    """Create weekday-conditioned transition matrices with Laplace smoothing."""
    transitions = np.full((7, n_clusters, n_clusters), laplace, dtype=float)
    for i in range(len(labels) - 1):
        wd = pd.Timestamp(dates[i]).weekday()
        transitions[wd, labels[i], labels[i + 1]] += 1.0
    transitions /= np.maximum(transitions.sum(axis=2, keepdims=True), 1e-12)
    return transitions


def cluster_members(daily: np.ndarray, labels: np.ndarray, n_clusters: int) -> list[np.ndarray]:
    """Collect historical day profiles for each cluster."""
    return [daily[labels == c] for c in range(n_clusters)]


def generate_series_weekday(
    days: int,
    centers: np.ndarray,
    weekday_transitions: np.ndarray,
    member_profiles: list[np.ndarray],
    stds: np.ndarray,
    start_cluster: int,
    start_date: pd.Timestamp,
    seed: int,
    noise_scale: float,
) -> np.ndarray:
    """Generate hourly synthetic series using weekday-conditioned transitions."""
    rng = np.random.default_rng(seed)
    cluster = int(start_cluster)
    current_date = pd.Timestamp(start_date)
    values: list[float] = []

    for _ in range(days):
        members = member_profiles[cluster]
        if len(members) > 0:
            day_profile = members[int(rng.integers(0, len(members)))].astype(float).copy()
        else:
            day_profile = centers[cluster].astype(float).copy()

        if noise_scale > 0:
            day_profile += rng.normal(0.0, stds[cluster] * noise_scale)
        day_profile = np.clip(day_profile, 0.0, None)
        values.extend(day_profile.tolist())

        wd = current_date.weekday()
        cluster = int(rng.choice(len(centers), p=weekday_transitions[wd, cluster]))
        current_date += pd.Timedelta(days=1)

    return np.asarray(values, dtype=float)


def prompt_positive_int(message: str, default: int) -> int:
    """Prompt user for a positive integer with default fallback."""
    while True:
        raw = input(f"{message} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Please enter a positive integer.")


def prompt_positive_float(message: str) -> float:
    """Prompt user for a positive float."""
    while True:
        raw = input(f"{message}: ").strip()
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Please enter a positive number.")


def main() -> None:
    """CLI entrypoint for 24x30 synthetic output generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic hourly load CSV.")
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY_PATH, help="Daily profiles .npy path")
    parser.add_argument("--hourly-input", type=Path, default=DEFAULT_HOURLY_PATH, help="Cleaned hourly CSV path")
    parser.add_argument("--kmeans", type=Path, default=DEFAULT_MODEL_PATH, help="KMeans model path")
    parser.add_argument("--markov", type=Path, default=DEFAULT_MARKOV_PATH, help="Markov transition path")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META_PATH, help="Metadata path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_PATH, help="Synthetic CSV output path")
    parser.add_argument("--days", type=int, default=None, help="Number of days to generate (prompted if omitted)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--noise-scale", type=float, default=0.03, help="Noise multiplier for variability")
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.05,
        help="Blend weight for synthetic profile vs historical hourly mean (0..1)",
    )
    parser.add_argument("--target-kwh", type=float, default=None, help="Target kWh input (monthly by default)")
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["monthly", "total"],
        default="monthly",
        help="Interpretation of --target-kwh: monthly scales with days/30, total is exact for full horizon",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2021-11-01",
        help="Output start date in YYYY-MM-DD format",
    )
    args = parser.parse_args()
    if not (0.0 <= args.blend_alpha <= 1.0):
        raise ValueError("--blend-alpha must be between 0 and 1")

    if args.days is None:
        print("Interactive Input")
        args.days = prompt_positive_int("Enter number of days", default=30)

    if args.target_kwh is None:
        args.target_kwh = prompt_positive_float("Enter monthly energy consumed (kWh)")

    if args.days <= 0:
        raise ValueError("--days must be positive")

    if args.target_kwh <= 0:
        raise ValueError("--target-kwh must be positive")

    daily = np.load(args.daily)
    hourly_df = pd.read_csv(args.hourly_input)
    if "x_Timestamp" not in hourly_df.columns:
        raise ValueError("Expected 'x_Timestamp' column in cleaned hourly CSV")
    hist_timestamps = pd.to_datetime(hourly_df["x_Timestamp"], errors="raise").to_numpy()

    model: KMeans = joblib.load(args.kmeans)
    transition = np.load(args.markov)

    k = int(model.n_clusters)
    if transition.shape != (k, k):
        raise ValueError(f"Transition shape must be ({k}, {k}), got {transition.shape}")

    transition = transition / np.maximum(transition.sum(axis=1, keepdims=True), 1e-12)

    labels = model.predict(daily)
    hist_days = min(len(labels), len(hist_timestamps) // 24)
    labels = labels[:hist_days]
    day_dates = hist_timestamps[: hist_days * 24 : 24]
    daily = daily[:hist_days]

    stds = cluster_hour_std(daily, labels, k)
    members = cluster_members(daily, labels, k)
    weekday_transitions = build_weekday_transitions(labels, day_dates, k, laplace=1.0)

    start_cluster = int(labels[-1])
    if args.meta.exists():
        with args.meta.open(encoding="utf-8") as f:
            meta = json.load(f)
        probs = np.asarray(meta.get("initial_cluster_probs", []), dtype=float)
        if probs.shape == (k,) and np.isfinite(probs).all() and probs.sum() > 0:
            probs = probs / probs.sum()
            start_cluster = int(np.random.default_rng(args.seed).choice(k, p=probs))

    series = generate_series_weekday(
        days=args.days,
        centers=model.cluster_centers_,
        weekday_transitions=weekday_transitions,
        member_profiles=members,
        stds=stds,
        start_cluster=start_cluster,
        start_date=pd.Timestamp(args.start_date),
        seed=args.seed,
        noise_scale=args.noise_scale,
    )

    # Blend with historical mean shape for stability while preserving stochastic patterns.
    hist_mean = np.mean(daily, axis=0)
    baseline = np.tile(hist_mean, args.days)
    series = args.blend_alpha * series + (1.0 - args.blend_alpha) * baseline

    target_total_kwh = float(args.target_kwh)
    if args.target_mode == "monthly":
        target_total_kwh = float(args.target_kwh) * (float(args.days) / 30.0)

    total = float(series.sum())
    if total > 0:
        series = series * (target_total_kwh / total)

    if len(series) != args.days * 24:
        raise RuntimeError("Generated series length mismatch")

    start_ts = pd.Timestamp(args.start_date)
    timestamps = pd.date_range(start=start_ts, periods=args.days * 24, freq="h")

    out = pd.DataFrame(
        {
            "x_Timestamp": timestamps,
            "t_kWh": series,
            "day": np.repeat(np.arange(1, args.days + 1), 24),
            "hour": np.tile(np.arange(24), args.days),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("Synthetic CSV generated")
    print(f"Rows: {len(out)}")
    print(f"Days: {args.days}")
    print(f"Output: {args.output}")
    print(f"Target mode: {args.target_mode}")
    print(f"Target total energy (kWh): {target_total_kwh:.3f}")
    print(f"Total energy (kWh): {out['t_kWh'].sum():.3f}")


if __name__ == "__main__":
    main()
