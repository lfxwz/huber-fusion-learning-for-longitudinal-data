"""Utilities for adding reproducible synthetic contamination."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def inject_outliers(
    responses: Sequence[np.ndarray],
    subject_fraction: float = 0.3,
    time_fraction: float = 1.0,
    noise_scale: float = 2.4,
    random_state: Any = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Return a contaminated copy of subject-level response vectors.

    Gaussian noise is added to randomly selected subjects and observation
    times. The input vectors are never modified in place.
    """
    if not 0.0 <= subject_fraction <= 1.0:
        raise ValueError("subject_fraction must be between 0 and 1.")
    if not 0.0 <= time_fraction <= 1.0:
        raise ValueError("time_fraction must be between 0 and 1.")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative.")

    contaminated = [np.asarray(values, dtype=float).reshape(-1).copy() for values in responses]
    n_subjects = len(contaminated)
    if n_subjects == 0:
        raise ValueError("responses must contain at least one subject.")

    rng = np.random.default_rng(random_state)
    n_selected = int(np.ceil(n_subjects * subject_fraction))
    if n_selected == 0 or time_fraction == 0.0:
        return contaminated, {
            "subject_indices": np.array([], dtype=int),
            "locations": {},
            "n_points": 0,
            "noise_scale": float(noise_scale),
        }

    subject_indices = np.sort(rng.choice(n_subjects, n_selected, replace=False))
    locations: dict[int, np.ndarray] = {}
    n_points = 0

    for subject_index in subject_indices:
        n_times = contaminated[subject_index].size
        n_selected_times = int(np.ceil(n_times * time_fraction))
        if n_selected_times == 0:
            continue

        time_indices = np.sort(rng.choice(n_times, n_selected_times, replace=False))
        contaminated[subject_index][time_indices] += rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=n_selected_times,
        )
        locations[int(subject_index)] = time_indices
        n_points += n_selected_times

    return contaminated, {
        "subject_indices": subject_indices,
        "locations": locations,
        "n_points": n_points,
        "noise_scale": float(noise_scale),
    }
