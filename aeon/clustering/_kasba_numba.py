from typing import Callable, Optional, Union

import numpy as np
from numba import njit
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering._k_means import EmptyClusterError
from aeon.clustering._kasba_utils import (
    _msm_from_multiple_to_multiple_distance,
    _msm_kesba_average,
    _msm_pairwise_distance,
)
from aeon.clustering.base import BaseClusterer
from aeon.distances import pairwise_distance
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic._msm import _msm_distance


class KASBA_NUMBA(BaseClusterer):
    """KASBA clusterer [1]
    KASBA is a $k$-means clustering algorithm designed for use with the MSM distance
    metric [2]. It finds initial clusters using an adapted form of kmeans++,
    uses a gradient descent to find barycentre averages and exploits the metric
    property of MSM to avoid distance calculations in assignment. It is very fast.
    """

    _tags = {
        "capability:multivariate": False,
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        distance: Union[str, Callable] = "msm",
        ba_subset_size: float = 0.5,
        initial_step_size: float = 0.05,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
        distance_params: Optional[dict] = None,
        count_distance_calls: bool = False,
        decay_rate: float = 0.1,
    ):
        self.distance = distance
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.initial_step_size = initial_step_size
        self.ba_subset_size = ba_subset_size
        self.count_distance_calls = count_distance_calls
        self.decay_rate = decay_rate
        self.n_clusters = n_clusters

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._distance_params = {}

        self.init_distance_calls = 0
        self.empty_cluster_distance_calls = 0
        self.update_distance_calls = 0
        self.assignment_distance_calls = 0
        self.total_distance_calls = 0
        super().__init__()

    def _fit(self, X: np.ndarray, y=None):
        self._check_params(X)

        cluster_centres, distances_to_centres, labels = self._elastic_kmeans_plus_plus(
            X,
        )

        self.labels_, self.cluster_centers_, self.inertia_, self.n_iter_ = _numba_kasba(
            X,
            n_clusters=self.n_clusters,
            cluster_centres=cluster_centres,
            distances_to_centres=distances_to_centres,
            labels=labels,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            c=self._distance_params.get("c", 1.0),
            independent=self._distance_params.get("independent", True),
            decay_rate=self.decay_rate,
            window=self._distance_params.get("window", None),
            ba_subset_size=self.ba_subset_size,
            initial_step_size=self.initial_step_size,
        )
        return self

    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        if isinstance(self.distance, str):
            pairwise_matrix = pairwise_distance(
                X, self.cluster_centers_, metric=self.distance, **self._distance_params
            )
        else:
            pairwise_matrix = pairwise_distance(
                X,
                self.cluster_centers_,
                metric=self.distance,
                **self._distance_params,
            )
        return pairwise_matrix.argmin(axis=1)

    def _elastic_kmeans_plus_plus(
        self,
        X,
    ):
        initial_center_idx = self._random_state.randint(X.shape[0])
        indexes = [initial_center_idx]

        mask = np.full((X.shape[0], 1), True)
        mask[initial_center_idx, 0] = False

        min_distances = pairwise_distance(
            X,
            X[initial_center_idx],
            metric=self.distance,
            mask=mask,
            **self._distance_params,
        ).flatten()
        self.init_distance_calls += len(X) - 1
        labels = np.zeros(X.shape[0], dtype=int)

        for i in range(1, self.n_clusters):
            probabilities = min_distances / min_distances.sum()
            next_center_idx = self._random_state.choice(X.shape[0], p=probabilities)
            indexes.append(next_center_idx)
            mask[next_center_idx, 0] = False

            new_distances = pairwise_distance(
                X,
                X[next_center_idx : next_center_idx + 1],
                metric=self.distance,
                mask=mask,
                **self._distance_params,
            ).flatten()
            self.init_distance_calls += len(X) - (i + 1)

            closer_points = new_distances < min_distances
            min_distances[closer_points] = new_distances[closer_points]
            labels[closer_points] = i

        centers = X[indexes]
        return centers, min_distances, labels

    def _create_numba_caches(self, X):
        self._check_params(X)
        size_of_subset = min(len(X), self.n_clusters * 2)
        subset = X[:size_of_subset]

        cluster_centres, distances_to_centres, labels = self._elastic_kmeans_plus_plus(
            subset,
        )

        _numba_kasba(
            subset,
            n_clusters=self.n_clusters,
            cluster_centres=cluster_centres,
            distances_to_centres=distances_to_centres,
            labels=labels,
            max_iter=2,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            c=self._distance_params.get("c", 1.0),
            independent=self._distance_params.get("independent", True),
            window=self._distance_params.get("window", None),
            decay_rate=self.decay_rate,
            ba_subset_size=self.ba_subset_size,
            initial_step_size=self.initial_step_size,
        )

    def _check_params(self, X: np.ndarray) -> None:
        self._random_state = check_random_state(self.random_state)

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than "
                f"n_cases ({X.shape[0]})"
            )

        self._distance_params = {
            **(self.distance_params or {}),
        }


@njit(cache=True, fastmath=True)
def _numba_kasba(
    X: np.ndarray,
    cluster_centres: np.ndarray,
    distances_to_centres: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    window: Optional[float],
    max_iter: int,
    tol: float,
    verbose: bool,
    random_state: int,
    c: float,
    independent: bool,
    decay_rate: float,
    ba_subset_size: float,
    initial_step_size: float,
):
    np.random.seed(random_state)

    x_size = X.shape[2]
    bounding_matrix = create_bounding_matrix(
        x_size, x_size, window=window, itakura_max_slope=None
    )

    inertia = np.inf
    prev_inertia = np.inf
    prev_labels = np.zeros(len(labels), dtype=np.int64)
    prev_cluster_centres = None
    for i in range(max_iter):

        cluster_centres, distances_to_centres = _recalculate_centroids(
            X=X,
            n_clusters=n_clusters,
            cluster_centres=cluster_centres,
            labels=labels,
            distances_to_centres=distances_to_centres,
            bounding_matrix=bounding_matrix,
            c=c,
            independent=independent,
            tol=tol,
            verbose=verbose,
            ba_subset_size=ba_subset_size,
            initial_step_size=initial_step_size,
            decay_rate=decay_rate,
        )

        labels, distances_to_centres, inertia = _fast_assign(
            X=X,
            n_clusters=n_clusters,
            cluster_centres=cluster_centres,
            distances_to_centres=distances_to_centres,
            labels=labels,
            is_first_iteration=(i == 0),
            c=c,
            independent=independent,
            bounding_matrix=bounding_matrix,
            verbose=verbose,
        )
        labels, cluster_centres, distances_to_centres = _handle_empty_cluster(
            X,
            n_clusters,
            cluster_centres,
            distances_to_centres,
            labels,
            c,
            independent,
            bounding_matrix,
        )

        if np.array_equal(prev_labels, labels):
            break

        prev_inertia = inertia
        prev_labels = labels.copy()
        prev_cluster_centres = cluster_centres.copy()

    if inertia < prev_inertia:
        return prev_labels, prev_cluster_centres, prev_inertia, i + 1
    return labels, cluster_centres, inertia, i + 1


@njit(cache=True, fastmath=True)
def _recalculate_centroids(
    X,
    n_clusters,
    cluster_centres,
    labels,
    distances_to_centres,
    bounding_matrix,
    c,
    independent,
    tol,
    verbose,
    ba_subset_size,
    initial_step_size,
    decay_rate,
):
    for j in range(n_clusters):
        current_cluster_indices = labels == j

        previous_distance_to_centre = distances_to_centres[current_cluster_indices]
        previous_cost = np.sum(previous_distance_to_centre)
        if len(X) <= 1:
            curr_centre = X[current_cluster_indices][0]
            dist_to_centre = np.zeros(1)
        else:
            curr_centre, dist_to_centre = _msm_kesba_average(
                X=X[current_cluster_indices],
                init_barycenter=cluster_centres[j],
                previous_cost=previous_cost,
                previous_distance_to_centre=previous_distance_to_centre,
                bounding_matrix=bounding_matrix,
                c=c,
                independent=independent,
                max_iters=50,
                tol=tol,
                verbose=verbose,
                ba_subset_size=ba_subset_size,
                initial_step_size=initial_step_size,
                decay_rate=decay_rate,
            )
        cluster_centres[j] = curr_centre
        distances_to_centres[current_cluster_indices] = dist_to_centre
    return cluster_centres, distances_to_centres


@njit(cache=True, fastmath=True)
def _fast_assign(
    X,
    n_clusters,
    cluster_centres,
    distances_to_centres,
    labels,
    is_first_iteration,
    c,
    independent,
    bounding_matrix,
    verbose,
):
    distances_between_centres = _msm_pairwise_distance(
        cluster_centres,
        c=c,
        independent=independent,
        bounding_matrix=bounding_matrix,
    )
    for i in range(X.shape[0]):
        min_dist = distances_to_centres[i]
        closest = labels[i]
        for j in range(n_clusters):
            if not is_first_iteration and j == closest:
                continue
            bound = distances_between_centres[j, closest] / 2.0
            if min_dist < bound:
                continue

            dist = _msm_distance(
                X[i],
                cluster_centres[j],
                bounding_matrix=bounding_matrix,
                independent=independent,
                c=c,
            )
            if dist < min_dist:
                min_dist = dist
                closest = j

        labels[i] = closest
        distances_to_centres[i] = min_dist

    inertia = np.sum(distances_to_centres**2)
    if verbose:
        print(f"{inertia} -->")
    return labels, distances_to_centres, inertia


@njit(cache=True, fastmath=True)
def _handle_empty_cluster(
    X: np.ndarray,
    n_clusters: int,
    cluster_centres: np.ndarray,
    distances_to_centres: np.ndarray,
    labels: np.ndarray,
    c,
    independent,
    bounding_matrix: np.ndarray,
):
    valid_clusters = np.arange(n_clusters)
    current_clusters = np.unique(labels)
    empty_clusters = set(valid_clusters) - set(current_clusters)
    j = 0
    if len(current_clusters) != n_clusters:
        print("Handling empty cluster")

        while len(empty_clusters) > 0:
            current_empty_cluster_index = empty_clusters.pop()
            index_furthest_from_centre = distances_to_centres.argmax()
            cluster_centres[current_empty_cluster_index] = X[index_furthest_from_centre]
            curr_pw = _msm_from_multiple_to_multiple_distance(
                X,
                cluster_centres,
                c=c,
                independent=independent,
                bounding_matrix=bounding_matrix,
            )

            distances_to_centres = np.zeros(len(distances_to_centres))
            labels = np.zeros(len(labels), dtype=np.int64)
            for i in range(len(curr_pw)):
                argmin_curr = curr_pw[i].argmin()
                min_val = curr_pw[i][argmin_curr]
                labels[i] = argmin_curr
                distances_to_centres[i] = min_val

            valid_clusters = np.arange(n_clusters)
            current_clusters = np.unique(labels)
            empty_clusters = set(valid_clusters) - set(current_clusters)
            j += 1
            if j > n_clusters:
                raise EmptyClusterError

    return labels, cluster_centres, distances_to_centres
