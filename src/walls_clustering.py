from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def allocate_centers_for_locations(counts: np.ndarray, k: int) -> np.ndarray:
    """
    Allocate k centers among distinct locations.

    Each location gets at least one center.
    Remaining centers are distributed proportionally to counts.
    """
    d = len(counts)
    n = counts.sum()

    if k < d:
        raise ValueError("This allocator is only for k >= number of distinct locations")

    allocation = np.ones(d, dtype=int)

    remaining = k - d
    if remaining == 0:
        return allocation

    quotas = remaining * counts / n
    extra = np.floor(quotas).astype(int)

    allocation += extra

    leftover = remaining - extra.sum()

    if leftover > 0:
        fractions = quotas - extra

        order = sorted(
            range(d),
            key=lambda i: (fractions[i], counts[i]),
            reverse=True,
        )

        for i in order[:leftover]:
            allocation[i] += 1

    return allocation


def cluster_points(
    points: np.ndarray,
    k: int,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster 2D points.

    Parameters
    ----------
    points:
        Array of shape (n, 2).
    k:
        Required number of clusters.
    random_state:
        Random seed.

    Returns
    -------
    centers:
        Array of shape (k, 2).
    labels:
        Array of shape (n,), where labels[i] is cluster id of points[i].
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")

    n = len(points)

    if n == 0:
        raise ValueError("points must be non-empty")

    if k <= 0:
        raise ValueError("k must be positive")

    rounded = np.rint(points).astype(np.int64)

    unique_positions, inverse, counts = np.unique(
        rounded,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    d = len(unique_positions)


    if k == d:
        centers = unique_positions.astype(float)
        labels = inverse.copy()
        return centers, labels


    if k < d:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

        model.fit(
            unique_positions.astype(float),
            sample_weight=counts,
        )

        location_labels = model.labels_
        labels = location_labels[inverse]
        centers = model.cluster_centers_

        return centers, labels


    allocation = allocate_centers_for_locations(counts, k)

    centers_list = []
    center_ids_by_location = []

    current_id = 0

    for position, amount in zip(unique_positions, allocation):
        ids = np.arange(current_id, current_id + amount)

        repeated_centers = np.repeat(
            position[None, :].astype(float),
            amount,
            axis=0,
        )

        centers_list.append(repeated_centers)
        center_ids_by_location.append(ids)

        current_id += amount

    centers = np.vstack(centers_list)

    labels = np.empty(n, dtype=int)
    rng = np.random.default_rng(random_state)


    for location_id, center_ids in enumerate(center_ids_by_location):
        point_indices = np.flatnonzero(inverse == location_id)

        rng.shuffle(point_indices)

        assigned = center_ids[
            np.arange(len(point_indices)) % len(center_ids)
        ]

        labels[point_indices] = assigned

    return centers, labels
