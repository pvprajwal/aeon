import numpy as np
from numba import njit

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._msm import (
    _msm_dependent_cost_matrix,
    _msm_distance,
    _msm_independent_cost_matrix,
)


@njit(cache=True, fastmath=True)
def _msm_pairwise_distance(
    X: np.ndarray,
    c: float,
    independent: bool,
    bounding_matrix: np.ndarray,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            distances[i, j] = _msm_distance(x1, x2, bounding_matrix, independent, c)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _msm_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    independent: bool,
    bounding_matrix: np.ndarray,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            distances[i, j] = _msm_distance(x1, y1, bounding_matrix, independent, c)
    return distances


@njit(cache=True, fastmath=True)
def _msm_from_single_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    independent: bool,
    bounding_matrix: np.ndarray,
) -> np.ndarray:
    n_cases = len(x)
    distances = np.zeros(n_cases)

    for i in range(n_cases):
        x1 = x[i]
        distances[i] = _msm_distance(x1, y, bounding_matrix, independent, c)
    return distances


@njit(cache=True, fastmath=True)
def _msm_kesba_average(
    X: np.ndarray,
    init_barycenter: np.ndarray,
    previous_cost: float,
    previous_distance_to_centre: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float = 1.0,
    independent: bool = True,
    max_iters: int = 50,
    tol=1e-5,
    verbose: bool = False,
    ba_subset_size: float = 0.5,
    initial_step_size: float = 0.05,
    decay_rate: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    X_size = X.shape[0]

    barycenter = np.copy(init_barycenter)
    prev_barycenter = np.copy(init_barycenter)

    distances_to_centre = np.zeros(X_size)
    num_ts_to_use = min(X_size, max(10, int(ba_subset_size * X_size)))
    if len(X) <= 1:
        distances_to_centre[0] = _msm_distance(
            X[0], barycenter, bounding_matrix, independent, c
        )
    else:
        for i in range(max_iters):
            shuffled_indices = np.random.permutation(X_size)
            if i > 0:
                shuffled_indices = shuffled_indices[:num_ts_to_use]

            current_step_size = initial_step_size * np.exp(-decay_rate * i)

            barycenter = _kasba_refine_one_iter(
                barycenter=barycenter,
                X=X,
                shuffled_indices=shuffled_indices,
                current_step_size=current_step_size,
                bounding_matrix=bounding_matrix,
                independent=independent,
                c=c,
            )

            pw_dist = _msm_from_single_to_multiple_distance(
                X, barycenter, c, independent, bounding_matrix
            )
            cost = np.sum(pw_dist)
            distances_to_centre = pw_dist.flatten()

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
                print(
                    f"[Subset-SSG-BA] epoch {i}, cost {f2s(cost, 5)}"
                )  # noqa: T001, T201

    return barycenter, distances_to_centre


@njit(cache=True, fastmath=True)
def f2s(f, precision=2):
    if np.isnan(f):
        return "NaN"
    s = str(int(np.floor(f))) + "."
    digits = f % 1
    for _ in range(precision):
        digits *= 10
        s += str(int(np.floor(digits)))
    return s


@njit(cache=True, fastmath=True)
def _kasba_refine_one_iter(
    barycenter: np.ndarray,
    X: np.ndarray,
    shuffled_indices: np.ndarray,
    current_step_size,
    bounding_matrix: np.ndarray,
    independent: bool = True,
    c: float = 1.0,
):

    X_size, X_dims, X_timepoints = X.shape

    barycenter_copy = np.copy(barycenter)

    for i in shuffled_indices:
        curr_ts = X[i]
        if independent:
            cost_matrix = _msm_independent_cost_matrix(
                curr_ts, barycenter_copy, bounding_matrix, c=c
            )
        else:
            cost_matrix = _msm_dependent_cost_matrix(
                curr_ts, barycenter_copy, bounding_matrix, c=c
            )

        curr_alignment = compute_min_return_path(cost_matrix)

        new_ba = np.zeros((X_dims, X_timepoints))
        for j, k in curr_alignment:
            new_ba[:, k] += barycenter_copy[:, k] - curr_ts[:, j]

        barycenter_copy -= (2.0 * current_step_size) * new_ba
    return barycenter_copy
