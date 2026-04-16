"""Train KMeans model on daily load profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "outputs" / "cleaned_hourly.csv"
DEFAULT_OUT_NPY = PROJECT_ROOT / "outputs" / "daily_profiles.npy"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "kmeans_model.pkl"
DEFAULT_META_PATH = Path(__file__).resolve().parent / "metadata.json"


def load_daily_profiles(data_path: Path) -> np.ndarray:
    """Load hourly CSV and reshape into daily 24-hour rows."""
    df = pd.read_csv(data_path)
    if "t_kWh" not in df.columns:
        raise ValueError("Expected 't_kWh' column in cleaned hourly CSV")

    values = pd.to_numeric(df["t_kWh"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    days = len(values) // 24
    if days < 2:
        raise ValueError("Not enough data: need at least 48 hourly points for clustering")
    return values[: days * 24].reshape(days, 24)


def choose_k(daily_profiles: np.ndarray, k_min: int, k_max: int, seed: int) -> int:
    """Choose cluster count by silhouette score."""
    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(daily_profiles)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(daily_profiles, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def train_and_save(
    daily_profiles: np.ndarray,
    n_clusters: int,
    seed: int,
    model_path: Path,
    metadata_path: Path,
    out_npy: Path,
) -> None:
    """Fit KMeans and persist artifacts."""
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    model.fit(daily_profiles)

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, daily_profiles)
    joblib.dump(model, model_path)

    metadata = {
        "clusters": n_clusters,
        "seed": seed,
        "days": int(daily_profiles.shape[0]),
        "inertia": float(model.inertia_),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    """CLI entrypoint for KMeans training."""
    parser = argparse.ArgumentParser(description="Train KMeans on daily load profiles.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Cleaned hourly CSV")
    parser.add_argument("--daily-out", type=Path, default=DEFAULT_OUT_NPY, help="Output .npy for daily profiles")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help="Output path for kmeans model")
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META_PATH, help="Output path for metadata")
    parser.add_argument("--k", type=int, default=6, help="Number of clusters (ignored when --auto-k is set)")
    parser.add_argument("--auto-k", action="store_true", help="Select K using silhouette score")
    parser.add_argument("--k-min", type=int, default=3, help="Minimum K for --auto-k")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum K for --auto-k")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    daily_profiles = load_daily_profiles(args.input)
    n_clusters = choose_k(daily_profiles, args.k_min, args.k_max, args.seed) if args.auto_k else args.k
    train_and_save(
        daily_profiles=daily_profiles,
        n_clusters=n_clusters,
        seed=args.seed,
        model_path=args.model_out,
        metadata_path=args.meta_out,
        out_npy=args.daily_out,
    )

    print("KMeans trained")
    print(f"Days: {daily_profiles.shape[0]}")
    print(f"Clusters: {n_clusters}")


if __name__ == "__main__":
    main()
