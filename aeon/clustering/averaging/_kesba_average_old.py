from typing import Optional, Union

import numpy as np
from numba import njit
from sklearn.utils import check_random_state

from aeon.clustering.averaging._ba_utils import (
    _get_alignment_path,
    _get_init_barycenter,
)
from aeon.distances import distance as distance_callable
from aeon.distances import pairwise_distance

# change stopping condition so if it increase for 10% of the max_iters in a row stop rather than just once


def _kesba_average(
    X: np.ndarray,
    distance: str = "dtw",
    max_iters: int = 30,
    tol=1e-5,
    init_barycenter: Union[np.ndarray, str] = "mean",
    precomputed_medoids_pairwise_distance: Optional[np.ndarray] = None,
    verbose: bool = False,
    random_state: Optional[int] = None,
    ba_subset_size: float = 1.0,
    return_distances: bool = False,
    count_number_distance_calls: bool = False,
    previous_cost: Optional[float] = None,
    previous_distance_to_centre: Optional[np.ndarray] = None,
    use_all_first_subset_ba_iteration: bool = False,
    lr_func: str = "iterative",
    initial_step_size: float = 0.05,
    decay_rate: float = 0.1,
    min_step_size: float = 0.005,
    **kwargs,
) -> np.ndarray:
    if len(X) <= 1:
        return X

    if X.ndim == 3:
        _X = X
    elif X.ndim == 2:
        _X = X.reshape((X.shape[0], 1, X.shape[1]))
    else:
        raise ValueError("X must be a 2D or 3D array")

    # Get the initial centre that is the mean
    barycenter = _get_init_barycenter(
        _X,
        init_barycenter,
        distance,
        precomputed_medoids_pairwise_distance,
        random_state,
        **kwargs,
    )

    random_state = check_random_state(random_state)
    X_size = _X.shape[0]

    if previous_cost is not None:
        cost_prev = previous_cost
    else:
        cost_prev = np.inf

    if previous_distance_to_centre is not None:
        prev_distances_to_centre = previous_distance_to_centre
    else:
        prev_distances_to_centre = np.zeros(X_size)

    prev_barycenter = np.copy(barycenter)

    cost = np.inf
    distances_to_centre = np.zeros(X_size)
    if distance == "wdtw" or distance == "wddtw":
        if "g" not in kwargs:
            kwargs["g"] = 0.05

    num_ts_to_use = min(X_size, max(10, int(ba_subset_size * X_size)))
    current_step_size = initial_step_size
    for i in range(max_iters):
        shuffled_indices = random_state.permutation(X_size)
        if i > 0 or not use_all_first_subset_ba_iteration:
            shuffled_indices = shuffled_indices[:num_ts_to_use]

        if lr_func == "iterative":
            current_step_size = initial_step_size / (i + 1)
        elif lr_func == "linear":
            current_step_size = initial_step_size * (1 - i / max_iters)
        elif lr_func == "quadratic":
            current_step_size = initial_step_size * (1 - (i / max_iters) ** 2)
        elif lr_func == "exponential":
            current_step_size = initial_step_size * np.exp(-decay_rate * i)
        elif lr_func == "dynamic-iterative":
            current_step_size = current_step_size / (i + 1)
        elif lr_func == "dynamic-linear":
            current_step_size = current_step_size * (1 - i / max_iters)
        elif lr_func == "dynamic-exponential":
            current_step_size = current_step_size * np.exp(-decay_rate * i)
        else:
            raise ValueError("Invalid learning rate function")

        barycenter = _ba_one_iter_random_subset_ssg(
            barycenter,
            _X,
            shuffled_indices,
            distance,
            current_step_size,
            **kwargs,
        )

        pw_dist = pairwise_distance(_X, barycenter, metric=distance, **kwargs)
        cost = np.sum(pw_dist)
        distances_to_centre = pw_dist.flatten()

        # Cost is the sum of distance to the centre
        if abs(cost_prev - cost) < tol:
            if cost_prev < cost:
                cost = cost_prev
                barycenter = prev_barycenter
                distances_to_centre = prev_distances_to_centre
            break
        elif cost_prev < cost:
            cost = cost_prev
            barycenter = prev_barycenter
            distances_to_centre = prev_distances_to_centre
            break

        prev_barycenter = barycenter
        prev_distances_to_centre = distances_to_centre.copy()
        cost_prev = cost

        if verbose:
            print(f"[Subset-SSG-BA] epoch {i}, cost {cost}")  # noqa: T001, T201

    if use_all_first_subset_ba_iteration:
        num_dist_computations = (i * num_ts_to_use) + X_size
    else:
        num_dist_computations = (i + 1) * num_ts_to_use

    if return_distances:
        if count_number_distance_calls:
            return barycenter, distances_to_centre, num_dist_computations
        return barycenter, distances_to_centre

    if count_number_distance_calls:
        return barycenter, num_dist_computations
    return barycenter


@njit(cache=True, fastmath=True)
def _ba_one_iter_random_subset_ssg(
    barycenter: np.ndarray,
    X: np.ndarray,
    shuffled_indices: np.ndarray,
    distance: str = "dtw",
    current_step_size: float = 0.05,
    window: Union[float, None] = None,
    g: float = 0.0,
    epsilon: Union[float, None] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 30,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
):

    X_size, X_dims, X_timepoints = X.shape
    # Only update current_step_size on the first iteration
    barycenter_copy = np.copy(barycenter)

    for i in shuffled_indices:
        curr_ts = X[i]
        curr_alignment, curr_cost = _get_alignment_path(
            barycenter_copy,
            X[i],
            distance,
            window,
            g,
            epsilon,
            nu,
            lmbda,
            independent,
            c,
            descriptor,
            reach,
            warp_penalty,
            transformation_precomputed,
            transformed_x,
            transformed_y,
        )

        new_ba = np.zeros((X_dims, X_timepoints))
        for j, k in curr_alignment:
            new_ba[:, k] += barycenter_copy[:, k] - curr_ts[:, j]

        barycenter_copy -= (2.0 * current_step_size) * new_ba
    return barycenter_copy


if __name__ == "__main__":
    from aeon.clustering.averaging import kasba_average
    from aeon.testing.data_generation import make_example_3d_numpy

    X_train = make_example_3d_numpy(20, 2, 10, random_state=1, return_y=False)
    distance = "dtw"

    holdit_ts = kasba_average(
        X_train,
        distance=distance,
        window=0.2,
        independent=False,
        method="holdit_stopping",
        holdit_num_ts_to_use_percentage=0.8,
    )
