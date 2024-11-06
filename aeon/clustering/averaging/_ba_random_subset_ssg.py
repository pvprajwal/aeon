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


def random_subset_ssg_barycenter_average(
    X: np.ndarray,
    distance: str = "dtw",
    max_iters: int = 30,
    tol=1e-5,
    init_barycenter: Union[np.ndarray, str] = "mean",
    initial_step_size: float = 0.05,
    final_step_size: float = 0.005,
    precomputed_medoids_pairwise_distance: Optional[np.ndarray] = None,
    verbose: bool = False,
    random_state: Optional[int] = None,
    ba_subset_size: float = 1.0,
    return_distances: bool = False,
    count_number_distance_calls: bool = False,
    previous_cost: Optional[float] = None,
    previous_distance_to_centre: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Compute the random subset ssg barycenter average of time series.

    Stochastic subgradient is much faster than petitjean, however, it is not guaranteed
    to find the optimal solution.
    This implements a stochastic subgradient DBA algorithm. This changes how
    the average is computed. Unlike traditional methods, it calculates a subgradient
    based on each individual time series within the dataset. The barycenter is then
    iteratively updated using these subgradient. See [2]_ for more details.

    Parameters
    ----------
    X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
            (n_cases, n_timepoints)
        A collection of time series instances to take the average from.
    distance: str, default='dtw'
        String defining the distance to use for averaging. Distance to
        compute similarity between time series. A list of valid strings for metrics
        can be found in the documentation form
        :func:`aeon.distances.get_distance_function`.
    max_iters: int, default=30
        Maximum number iterations for dba to update over.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the Expectation-Maximization procedure stops.
    init_barycenter: np.ndarray or, default=None
        The initial barycenter to use for the minimisation. If a np.ndarray is provided
        it must be of shape ``(n_channels, n_timepoints)``. If a str is provided it must
        be one of the following: ['mean', 'medoids', 'random'].
    initial_step_size : float (default: 0.05)
        Initial step size for the subgradient descent algorithm.
        Default value is suggested by [2]_.
    final_step_size : float (default: 0.005)
        Final step size for the subgradient descent algorithm.
        Default value is suggested by [2]_.
    weights: Optional[np.ndarray] of shape (n_cases,), default=None
        The weights associated to each time series instance, if None a weight
        of 1 will be associated to each instance.
    precomputed_medoids_pairwise_distance: np.ndarray (of shape (len(X), len(X)),
                default=None
        Precomputed medoids pairwise.
    verbose: bool, default=False
        Boolean that controls the verbosity.
    random_state: int or None, default=None
        Random state to use for the barycenter averaging.
    ba_subset_size: float, default=1.0
        The proportion of the dataset to use for the barycenter averaging. If set to 1.0
        the full dataset will be used.
    **kwargs
        Keyword arguments to pass to the distance metric.

    Returns
    -------
    C: np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the average of the collection of instances provided.
    p: np.ndarray of shape (n_cases,) of distances between X and C


    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
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

    current_step_size = initial_step_size
    num_ts_to_use = min(X_size, max(10, int(ba_subset_size * X_size)))
    # Loop up to 30 times
    for i in range(max_iters):
        # Randomly order the dataset
        shuffled_indices = random_state.permutation(X_size)[:num_ts_to_use]
        # It then warps all onto centre to get the Fretchet distance
        # Updating the barycenter every iteration based on the warping
        barycenter, current_step_size = _ba_one_iter_random_subset_ssg(
            barycenter,
            _X,
            shuffled_indices,
            distance,
            initial_step_size,
            final_step_size,
            current_step_size,
            i,
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

    if verbose:
        print(
            f"[Subset-SSG-BA] finished in {i + 1} iterations, cost {cost}"
        )  # noqa: T001, T201

    if return_distances:
        if count_number_distance_calls:
            return barycenter, distances_to_centre, (i + 1) * num_ts_to_use
        return barycenter, distances_to_centre

    if count_number_distance_calls:
        return barycenter, (i + 1) * num_ts_to_use
    return barycenter


@njit(cache=True, fastmath=True)
def _ba_one_iter_random_subset_ssg(
    barycenter: np.ndarray,
    X: np.ndarray,
    shuffled_indices: np.ndarray,
    distance: str = "dtw",
    initial_step_size: float = 0.05,
    final_step_size: float = 0.005,
    current_step_size: float = 0.05,
    iteration: int = 0,
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
    step_size_reduction = 0.0
    if iteration == 0:
        step_size_reduction = (initial_step_size - final_step_size) / X_size

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

        current_step_size -= step_size_reduction
    return barycenter_copy, current_step_size


if __name__ == "__main__":
    from aeon.clustering.averaging import elastic_barycenter_average
    from aeon.testing.data_generation import make_example_3d_numpy

    X_train = make_example_3d_numpy(20, 2, 10, random_state=1, return_y=False)
    distance = "dtw"

    holdit_ts = elastic_barycenter_average(
        X_train,
        distance=distance,
        window=0.2,
        independent=False,
        method="holdit_stopping",
        holdit_num_ts_to_use_percentage=0.8,
    )
