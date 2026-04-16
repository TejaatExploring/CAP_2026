"""Build day-to-day Markov transitions over KMeans daily clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAILY_PATH = PROJECT_ROOT / "outputs" / "daily_profiles.npy"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "kmeans_model.pkl"
DEFAULT_META_PATH = Path(__file__).resolve().parent / "metadata.json"
DEFAULT_OUT_PATH = Path(__file__).resolve().parent / "markov_transition.npy"


def build_transition(labels: np.ndarray, n_clusters: int, laplace: float = 1.0) -> np.ndarray:
    """Create a row-stochastic day-to-day transition matrix."""
    counts = np.full((n_clusters, n_clusters), laplace, dtype=float)
    for i in range(len(labels) - 1):
        counts[labels[i], labels[i + 1]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def main() -> None:
    """CLI entrypoint for Markov transition matrix creation."""
    parser = argparse.ArgumentParser(description="Build Markov transitions for daily clusters.")
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY_PATH, help="Daily profiles .npy path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Trained KMeans model path")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META_PATH, help="Metadata path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Transition output .npy path")
    parser.add_argument("--laplace", type=float, default=1.0, help="Laplace smoothing value")
    args = parser.parse_args()

    daily = np.load(args.daily)
    kmeans = joblib.load(args.model)
    labels = kmeans.predict(daily)
    n_clusters = int(kmeans.n_clusters)

    transition = build_transition(labels, n_clusters, laplace=args.laplace)
    np.save(args.out, transition)

    if args.meta.exists():
        with args.meta.open(encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata.update(
        {
            "clusters": n_clusters,
            "markov_laplace": args.laplace,
            "markov_train_days": int(len(labels)),
            "initial_cluster_probs": (
                (np.bincount(labels, minlength=n_clusters) / len(labels)).astype(float).tolist()
            ),
        }
    )
    with args.meta.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Markov transition matrix saved")
    print(f"Shape: {transition.shape}")


if __name__ == "__main__":
    main()
