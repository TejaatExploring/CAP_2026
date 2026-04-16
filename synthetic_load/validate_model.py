"""Validate synthetic load model with leakage-free holdout evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "outputs" / "cleaned_hourly.csv"
DEFAULT_KMEANS_PATH = Path(__file__).resolve().parent / "kmeans_model.pkl"
DEFAULT_MARKOV_PATH = Path(__file__).resolve().parent / "markov_transition.npy"
DEFAULT_META_PATH = Path(__file__).resolve().parent / "metadata.json"


def safe_mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-6) -> float:
    """Compute MAPE robustly when actual contains zeros."""
    denom = np.maximum(np.abs(actual), eps)
    return float(np.mean(np.abs((actual - predicted) / denom)) * 100.0)


def split_daily(values: np.ndarray, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Split hourly series into daily train and validation matrices."""
    days = len(values) // 24
    if days < 14:
        raise ValueError("Need at least 14 full days for meaningful validation")
    daily = values[: days * 24].reshape(days, 24)

    train_days = max(7, int(days * train_ratio))
    train_days = min(train_days, days - 1)
    return daily[:train_days], daily[train_days:]


def split_daily_with_dates(
    values: np.ndarray,
    timestamps: np.ndarray,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split hourly series and associated day-start timestamps into train/validation."""
    days = len(values) // 24
    if days < 14:
        raise ValueError("Need at least 14 full days for meaningful validation")

    values = values[: days * 24]
    timestamps = timestamps[: days * 24]

    daily = values.reshape(days, 24)
    day_starts = timestamps[::24]

    train_days = max(7, int(days * train_ratio))
    train_days = min(train_days, days - 1)
    return daily[:train_days], daily[train_days:], day_starts[:train_days], day_starts[train_days:]


def build_transition(labels: np.ndarray, n_clusters: int, laplace: float = 1.0) -> np.ndarray:
    """Build smoothed day-to-day cluster transition matrix."""
    counts = np.full((n_clusters, n_clusters), laplace, dtype=float)
    for i in range(len(labels) - 1):
        counts[labels[i], labels[i + 1]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def cluster_hour_std(train_daily: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Estimate per-cluster hourly variability for stochastic generation."""
    stds = np.zeros((n_clusters, 24), dtype=float)
    global_std = np.std(train_daily, axis=0)
    global_std = np.maximum(global_std, 1e-3)
    for c in range(n_clusters):
        members = train_daily[labels == c]
        if len(members) < 2:
            stds[c] = global_std
        else:
            stds[c] = np.maximum(np.std(members, axis=0), 1e-3)
    return stds


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


def cluster_members(train_daily: np.ndarray, labels: np.ndarray, n_clusters: int) -> list[np.ndarray]:
    """Collect daily profiles per cluster for bootstrap-style sampling."""
    members: list[np.ndarray] = []
    for c in range(n_clusters):
        members.append(train_daily[labels == c])
    return members


def generate_synthetic(
    n_days: int,
    centers: np.ndarray,
    transition: np.ndarray,
    stds: np.ndarray,
    start_cluster: int,
    seed: int,
    noise_scale: float,
) -> np.ndarray:
    """Generate synthetic hourly series by sampling daily clusters."""
    rng = np.random.default_rng(seed)
    cluster = int(start_cluster)
    generated: list[float] = []

    for _ in range(n_days):
        noise = rng.normal(0.0, stds[cluster] * noise_scale)
        day_profile = np.clip(centers[cluster] + noise, 0.0, None)
        generated.extend(day_profile.tolist())
        cluster = int(rng.choice(len(centers), p=transition[cluster]))

    return np.asarray(generated, dtype=float)


def generate_synthetic_weekday(
    n_days: int,
    centers: np.ndarray,
    weekday_transitions: np.ndarray,
    member_profiles: list[np.ndarray],
    stds: np.ndarray,
    start_cluster: int,
    start_date: pd.Timestamp,
    seed: int,
    noise_scale: float,
) -> np.ndarray:
    """Generate synthetic series using weekday-conditioned transitions and profile sampling."""
    rng = np.random.default_rng(seed)
    cluster = int(start_cluster)
    current_date = pd.Timestamp(start_date)
    generated: list[float] = []

    for _ in range(n_days):
        members = member_profiles[cluster]
        if len(members) > 0:
            day_profile = members[int(rng.integers(0, len(members)))].astype(float).copy()
        else:
            day_profile = centers[cluster].astype(float).copy()

        if noise_scale > 0:
            day_profile += rng.normal(0.0, stds[cluster] * noise_scale)
        day_profile = np.clip(day_profile, 0.0, None)
        generated.extend(day_profile.tolist())

        wd = current_date.weekday()
        probs = weekday_transitions[wd, cluster]
        cluster = int(rng.choice(len(centers), p=probs))
        current_date += pd.Timedelta(days=1)

    return np.asarray(generated, dtype=float)


def parse_noise_grid(noise_grid: str) -> list[float]:
    """Parse comma-separated noise scales into a list of floats."""
    values = []
    for token in noise_grid.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Noise grid is empty")
    return sorted(set(values))


def hourly_calibration_factors(actual_train: np.ndarray, synthetic_train: np.ndarray) -> np.ndarray:
    """Create hourly correction factors using training data only."""
    actual_hourly = np.array([actual_train[h::24].mean() for h in range(24)], dtype=float)
    synth_hourly = np.array([synthetic_train[h::24].mean() for h in range(24)], dtype=float)
    factors = np.divide(actual_hourly, np.maximum(synth_hourly, 1e-6))
    return np.clip(factors, 0.6, 1.4)


def apply_hourly_factors(series: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Apply 24-hour correction factors to a flat hourly series."""
    out = np.asarray(series, dtype=float).copy()
    for i in range(len(out)):
        out[i] *= factors[i % 24]
    return out


def build_weekday_daily_pools(train_daily: np.ndarray, train_dates: np.ndarray) -> dict[int, np.ndarray]:
    """Build weekday->daily total pools from training data."""
    totals = train_daily.sum(axis=1)
    pools: dict[int, list[float]] = {i: [] for i in range(7)}
    for i, total in enumerate(totals):
        wd = pd.Timestamp(train_dates[i]).weekday()
        pools[wd].append(float(total))
    return {k: np.asarray(v, dtype=float) for k, v in pools.items()}


def sample_daily_totals_for_dates(
    dates: np.ndarray,
    pools: dict[int, np.ndarray],
    seed: int,
) -> np.ndarray:
    """Sample target daily totals by weekday from training pools."""
    rng = np.random.default_rng(seed)
    all_totals = np.concatenate([v for v in pools.values() if len(v) > 0])
    fallback = float(np.mean(all_totals)) if len(all_totals) else 1.0

    sampled = []
    for d in dates:
        wd = pd.Timestamp(d).weekday()
        options = pools.get(wd, np.array([], dtype=float))
        if len(options) == 0:
            sampled.append(fallback)
        else:
            sampled.append(float(options[int(rng.integers(0, len(options)))]))
    return np.asarray(sampled, dtype=float)


def apply_daily_total_targets(series: np.ndarray, daily_targets: np.ndarray) -> np.ndarray:
    """Scale each generated day to match sampled daily energy targets."""
    if len(series) != len(daily_targets) * 24:
        raise ValueError("Series length and daily target count are inconsistent")

    reshaped = series.reshape(len(daily_targets), 24).copy()
    for i in range(len(daily_targets)):
        cur = float(np.sum(reshaped[i]))
        if cur > 0:
            reshaped[i] *= float(daily_targets[i] / cur)
    return reshaped.reshape(-1)


def tune_blend_alpha(train_actual: np.ndarray, synthetic_train: np.ndarray, baseline_train: np.ndarray) -> float:
    """Tune blend alpha on training data only to minimize RMSE."""
    best_alpha = 1.0
    best_rmse = float("inf")
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = alpha * synthetic_train + (1.0 - alpha) * baseline_train
        rmse = float(np.sqrt(mean_squared_error(train_actual, blended)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
    return best_alpha


def build_flat_baseline(train_daily: np.ndarray, n_days: int) -> np.ndarray:
    """Flat baseline repeating mean hourly profile from training data."""
    hourly_mean = np.mean(train_daily, axis=0)
    return np.tile(hourly_mean, n_days)


def main() -> None:
    """CLI entrypoint for leakage-free model validation."""
    parser = argparse.ArgumentParser(description="Validate synthetic load generation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Cleaned hourly CSV")
    parser.add_argument("--kmeans", type=Path, default=DEFAULT_KMEANS_PATH, help="Existing KMeans model path")
    parser.add_argument("--markov", type=Path, default=DEFAULT_MARKOV_PATH, help="Existing Markov transition path")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META_PATH, help="Metadata path with initial probabilities")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of days used for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--noise-scale", type=float, default=0.08, help="Noise multiplier for variability")
    parser.add_argument(
        "--noise-grid",
        type=str,
        default="0.00,0.03,0.05,0.08,0.10,0.12",
        help="Comma-separated noise scales to tune on train split",
    )
    parser.add_argument("--disable-tuning", action="store_true", help="Disable train-split noise tuning")
    parser.add_argument("--no-plot", action="store_true", help="Disable comparison plot")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "t_kWh" not in df.columns:
        raise ValueError("Expected 't_kWh' column in cleaned hourly CSV")
    if "x_Timestamp" not in df.columns:
        raise ValueError("Expected 'x_Timestamp' column in cleaned hourly CSV")

    values = pd.to_numeric(df["t_kWh"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    timestamps = pd.to_datetime(df["x_Timestamp"], errors="raise").to_numpy()
    finite_mask = np.isfinite(pd.to_numeric(df["t_kWh"], errors="coerce").to_numpy(dtype=float))
    timestamps = timestamps[finite_mask]

    train_daily, val_daily, train_dates, val_dates = split_daily_with_dates(values, timestamps, args.train_ratio)

    model: KMeans = joblib.load(args.kmeans)
    k = int(model.n_clusters)
    train_labels = model.predict(train_daily)
    member_profiles = cluster_members(train_daily, train_labels, k)
    weekday_transitions = build_weekday_transitions(train_labels, train_dates, k, laplace=1.0)

    transition_artifact = np.load(args.markov)
    if transition_artifact.shape != (k, k):
        transition = build_transition(train_labels, k, laplace=1.0)
    else:
        transition = transition_artifact / np.maximum(transition_artifact.sum(axis=1, keepdims=True), 1e-12)

    start_cluster = int(train_labels[-1])
    if args.meta.exists():
        with args.meta.open(encoding="utf-8") as f:
            meta = json.load(f)
        probs = np.asarray(meta.get("initial_cluster_probs", []), dtype=float)
        if probs.shape == (k,) and np.isfinite(probs).all() and probs.sum() > 0:
            probs = probs / probs.sum()
            start_cluster = int(np.random.default_rng(args.seed).choice(k, p=probs))

    stds = cluster_hour_std(train_daily, train_labels, k)
    train_actual = train_daily.reshape(-1)

    chosen_noise = args.noise_scale
    if not args.disable_tuning:
        grid = parse_noise_grid(args.noise_grid)
        best_rmse = float("inf")
        for noise in grid:
            synthetic_train = generate_synthetic_weekday(
                n_days=train_daily.shape[0],
                centers=model.cluster_centers_,
                weekday_transitions=weekday_transitions,
                member_profiles=member_profiles,
                stds=stds,
                start_cluster=start_cluster,
                start_date=pd.Timestamp(train_dates[0]),
                seed=args.seed,
                noise_scale=noise,
            )
            rmse_noise = float(np.sqrt(mean_squared_error(train_actual, synthetic_train)))
            if rmse_noise < best_rmse:
                best_rmse = rmse_noise
                chosen_noise = noise

    synthetic_train = generate_synthetic_weekday(
        n_days=train_daily.shape[0],
        centers=model.cluster_centers_,
        weekday_transitions=weekday_transitions,
        member_profiles=member_profiles,
        stds=stds,
        start_cluster=start_cluster,
        start_date=pd.Timestamp(train_dates[0]),
        seed=args.seed,
        noise_scale=chosen_noise,
    )
    factors = hourly_calibration_factors(train_actual, synthetic_train)
    synthetic_train = apply_hourly_factors(synthetic_train, factors)

    train_daily_pools = build_weekday_daily_pools(train_daily, train_dates)
    train_targets = sample_daily_totals_for_dates(train_dates, train_daily_pools, seed=args.seed + 7)
    synthetic_train = apply_daily_total_targets(synthetic_train, train_targets)

    synthetic = generate_synthetic_weekday(
        n_days=val_daily.shape[0],
        centers=model.cluster_centers_,
        weekday_transitions=weekday_transitions,
        member_profiles=member_profiles,
        stds=stds,
        start_cluster=start_cluster,
        start_date=pd.Timestamp(val_dates[0]),
        seed=args.seed,
        noise_scale=chosen_noise,
    )
    synthetic = apply_hourly_factors(synthetic, factors)

    val_targets = sample_daily_totals_for_dates(val_dates, train_daily_pools, seed=args.seed + 17)
    synthetic = apply_daily_total_targets(synthetic, val_targets)

    actual = val_daily.reshape(-1)
    baseline = build_flat_baseline(train_daily, val_daily.shape[0])
    baseline_train = build_flat_baseline(train_daily, train_daily.shape[0])

    blend_alpha = tune_blend_alpha(train_actual, synthetic_train, baseline_train)
    synthetic = blend_alpha * synthetic + (1.0 - blend_alpha) * baseline

    mape = safe_mape(actual, synthetic)
    rmse = float(np.sqrt(mean_squared_error(actual, synthetic)))
    baseline_mape = safe_mape(actual, baseline)
    baseline_rmse = float(np.sqrt(mean_squared_error(actual, baseline)))

    print("\n===== HOLDOUT ERROR METRICS =====")
    print(f"Validation days: {val_daily.shape[0]}")
    print(f"Chosen noise scale: {chosen_noise:.3f}")
    print(f"Blend alpha (synthetic weight): {blend_alpha:.2f}")
    print(f"Synthetic MAPE: {mape:.2f} %")
    print(f"Synthetic RMSE: {rmse:.3f} kWh")
    print(f"Baseline MAPE : {baseline_mape:.2f} %")
    print(f"Baseline RMSE : {baseline_rmse:.3f} kWh")

    if not args.no_plot:
        horizon = min(168, len(actual))
        x = np.arange(horizon)
        plt.figure(figsize=(13, 5))
        plt.plot(x, actual[:horizon], label="Actual", linewidth=2)
        plt.plot(x, synthetic[:horizon], "--", label="Synthetic", linewidth=2)
        plt.plot(x, baseline[:horizon], ":", label="Flat baseline", linewidth=2)
        plt.xlabel("Hour")
        plt.ylabel("Energy (kWh)")
        plt.title("Holdout Validation: Actual vs Synthetic vs Baseline")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
