from typing import Optional

import numpy as np
from numba import njit
from sklearn.utils import check_random_state

from aeon.distances import msm_alignment_path, pairwise_distance, twe_alignment_path


def kasba_average(
    X: np.ndarray,
    init_barycenter: np.ndarray,
    previous_cost: float,
    previous_distance_to_centre: np.ndarray,
    distance: str = "msm",
    max_iters: int = 50,
    tol=1e-5,
    verbose: bool = False,
    random_state: Optional[int] = None,
    ba_subset_size: float = 0.5,
    initial_step_size: float = 0.05,
    decay_rate: float = 0.1,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(X) <= 1:
        return X, np.zeros(X.shape[0]), 0

    if X.ndim == 3:
        _X = X
    elif X.ndim == 2:
        _X = X.reshape((X.shape[0], 1, X.shape[1]))
    else:
        raise ValueError("X must be a 2D or 3D array")

    random_state = check_random_state(random_state)
    X_size = _X.shape[0]

    barycenter = np.copy(init_barycenter)
    prev_barycenter = np.copy(init_barycenter)

    distances_to_centre = np.zeros(X_size)
    num_ts_to_use = min(X_size, max(10, int(ba_subset_size * X_size)))
    for i in range(max_iters):
        shuffled_indices = random_state.permutation(X_size)
        if i > 0:
            shuffled_indices = shuffled_indices[:num_ts_to_use]

        current_step_size = initial_step_size * np.exp(-decay_rate * i)

        barycenter = _kasba_refine_one_iter(
            barycenter=barycenter,
            X=_X,
            shuffled_indices=shuffled_indices,
            current_step_size=current_step_size,
            distance=distance,
            **kwargs,
        )

        pw_dist = pairwise_distance(_X, barycenter, metric=distance, **kwargs)
        cost = np.sum(pw_dist)
        distances_to_centre = pw_dist.flatten()

        # Cost is the sum of distance to the centre
        if abs(previous_cost - cost) < tol:
            if previous_cost < cost:
                barycenter = prev_barycenter
                distances_to_centre = previous_distance_to_centre
            break
        elif previous_cost < cost:
            barycenter = prev_barycenter
            distances_to_centre = previous_distance_to_centre
            break

        prev_barycenter = barycenter
        previous_distance_to_centre = distances_to_centre.copy()
        previous_cost = cost

        if verbose:
            print(f"[Subset-SSG-BA] epoch {i}, cost {cost}")  # noqa: T001, T201

    num_dist_computations = (i * num_ts_to_use) + X_size

    return barycenter, distances_to_centre, num_dist_computations


@njit(cache=True, fastmath=True)
def _kasba_refine_one_iter(
    barycenter: np.ndarray,
    X: np.ndarray,
    shuffled_indices: np.ndarray,
    current_step_size,
    distance: str = "msm",
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
):

    X_size, X_dims, X_timepoints = X.shape

    barycenter_copy = np.copy(barycenter)

    for i in shuffled_indices:
        curr_ts = X[i]
        if distance == "twe":
            curr_alignment, curr_cost = twe_alignment_path(
                curr_ts, barycenter_copy, nu=nu, lmbda=lmbda
            )
        elif distance == "msm":
            curr_alignment, curr_cost = msm_alignment_path(
                curr_ts, barycenter_copy, independent=independent, c=c
            )
        else:
            raise ValueError(f"Invalid distance metric: {distance}")

        new_ba = np.zeros((X_dims, X_timepoints))
        for j, k in curr_alignment:
            new_ba[:, k] += barycenter_copy[:, k] - curr_ts[:, j]

        barycenter_copy -= (2.0 * current_step_size) * new_ba
    return barycenter_copy
