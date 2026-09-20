from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons


def synthetic_prior_shapes(seed: int = 42, n_samples: int = 320) -> dict[str, pd.DataFrame]:
    """Create deterministic educational shapes; these are not model training artifacts."""
    rng = np.random.default_rng(seed)

    blobs, blob_labels = make_blobs(
        n_samples=n_samples,
        centers=4,
        cluster_std=(0.55, 0.8, 0.45, 0.7),
        random_state=seed,
    )
    moons, moon_labels = make_moons(n_samples=n_samples, noise=0.10, random_state=seed)
    rings, ring_labels = make_circles(
        n_samples=n_samples, factor=0.42, noise=0.075, random_state=seed
    )

    latent = rng.normal(size=n_samples)
    switch = rng.integers(0, 2, size=n_samples)
    scm_x = latent + rng.normal(0, 0.28, n_samples)
    scm_y = np.where(switch, 0.7 * latent**2 - 0.5, 0.9 * latent) + rng.normal(
        0, 0.25, n_samples
    )

    half = n_samples // 2
    angle = np.sqrt(rng.random(half)) * 3.5 * np.pi
    radius = 0.30 * angle
    arm_x = radius * np.cos(angle) + rng.normal(0, 0.18, half)
    arm_y = radius * np.sin(angle) + rng.normal(0, 0.18, half)
    spiral_x = np.concatenate([arm_x, -arm_x])
    spiral_y = np.concatenate([arm_y, -arm_y])
    spiral_labels = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])

    return {
        "Clusters": _frame(blobs[:, 0], blobs[:, 1], blob_labels),
        "Hybrid SCM": _frame(scm_x, scm_y, switch),
        "Two moons": _frame(moons[:, 0], moons[:, 1], moon_labels),
        "Rings": _frame(rings[:, 0], rings[:, 1], ring_labels),
        "Spirals": _frame(spiral_x, spiral_y, spiral_labels),
    }


def _frame(x: np.ndarray, y: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": x, "y": y, "pattern": group.astype(str)})
