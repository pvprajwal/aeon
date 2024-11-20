"""Time series kmeans."""

from typing import Optional

__maintainer__ = []

from typing import Callable, Union

import numpy as np
from numpy.f2py.crackfortran import verbose
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering._k_means import EmptyClusterError
from aeon.clustering.averaging import kasba_average
from aeon.clustering.base import BaseClusterer
from aeon.distances import distance as distance_func
from aeon.distances import pairwise_distance


class KASBA(BaseClusterer):
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
        final_step_size: float = 0.005,
        window: float = 0.5,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
        distance_params: Optional[dict] = None,
        average_method: str = "lr_random_subset_ssg",
        count_distance_calls: bool = False,
        use_mean_as_init: bool = False,
        use_previous_cost: bool = True,
        use_all_first_subset_ba_iteration: bool = True,
        ba_lr_func: str = "exponential",
        decay_rate: float = 0.1,
    ):
        self.distance = distance
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.initial_step_size = initial_step_size
        self.final_step_size = final_step_size
        self.window = window
        self.ba_subset_size = ba_subset_size
        self.average_method = average_method
        self.count_distance_calls = count_distance_calls
        self.use_mean_as_init = use_mean_as_init
        self.use_previous_cost = use_previous_cost
        self.use_all_first_subset_ba_iteration = use_all_first_subset_ba_iteration
        self.ba_lr_func = ba_lr_func
        self.decay_rate = decay_rate

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
        super().__init__(n_clusters)

    def _fit(self, X: np.ndarray, y=None):
        self._check_params(X)
        cluster_centres, distances_to_centres, labels = self._elastic_kmeans_plus_plus(
            X,
        )
        self.labels_, self.cluster_centers_, self.inertia_, self.n_iter_ = self._kesba(
            X,
            cluster_centres,
            distances_to_centres,
            labels,
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

    def _kesba(
        self,
        X,
        cluster_centres,
        distances_to_centres,
        labels,
    ):
        inertia = np.inf
        prev_inertia = np.inf
        prev_labels = None
        prev_cluster_centres = None
        for i in range(self.max_iter):

            cluster_centres, distances_to_centres = self._recalculate_centroids(
                X,
                cluster_centres,
                labels,
                distances_to_centres,
            )

            labels, distances_to_centres, inertia = self._fast_assign(
                X,
                cluster_centres,
                distances_to_centres,
                labels,
                i == 0,
            )
            labels, cluster_centres, distances_to_centres = self._handle_empty_cluster(
                X,
                cluster_centres,
                distances_to_centres,
                labels,
            )

            if np.array_equal(prev_labels, labels):
                break

            prev_inertia = inertia
            prev_labels = labels.copy()
            prev_cluster_centres = cluster_centres.copy()
        if inertia < prev_inertia:
            return prev_labels, prev_cluster_centres, prev_inertia, i + 1
        return labels, cluster_centres, inertia, i + 1

    def _fast_assign(
        self,
        X,
        cluster_centres,
        distances_to_centres,
        labels,
        is_first_iteration,
    ):
        distances_between_centres = pairwise_distance(
            cluster_centres,
            cluster_centres,
            metric=self.distance,
            **self._distance_params,
        )
        self.assignment_distance_calls += len(cluster_centres) * len(cluster_centres)
        for i in range(X.shape[0]):
            min_dist = distances_to_centres[i]
            closest = labels[i]
            for j in range(self.n_clusters):
                if not is_first_iteration and j == closest:
                    continue
                bound = distances_between_centres[j, closest] / 2.0
                if min_dist < bound:
                    continue

                dist = distance_func(
                    X[i],
                    cluster_centres[j],
                    metric=self.distance,
                    **self._distance_params,
                )
                self.assignment_distance_calls += 1
                if dist < min_dist:
                    min_dist = dist
                    closest = j

            labels[i] = closest
            distances_to_centres[i] = min_dist

        inertia = np.sum(distances_to_centres**2)
        if self.verbose:
            print(f"{inertia:.5f}", end=" --> ")
        return labels, distances_to_centres, inertia

    def _recalculate_centroids(
        self,
        X,
        cluster_centres,
        labels,
        distances_to_centres,
    ):

        for j in range(self.n_clusters):
            previous_distance_to_centre = distances_to_centres[labels == j]
            previous_cost = np.sum(previous_distance_to_centre)
            curr_centre, dist_to_centre, num_distance_calls = kasba_average(
                X[labels == j],
                max_iters=50,
                method=self.average_method,
                init_barycenter=cluster_centres[j],
                distance=self.distance,
                initial_step_size=self.initial_step_size,
                final_step_size=self.final_step_size,
                random_state=self._random_state,
                return_distances=True,
                count_number_distance_calls=True,
                verbose=self.verbose,
                ba_subset_size=self.ba_subset_size,
                previous_cost=previous_cost,
                previous_distance_to_centre=previous_distance_to_centre,
                use_all_first_subset_ba_iteration=self.use_all_first_subset_ba_iteration,
                lr_func=self.ba_lr_func,
                decay_rate=self.decay_rate,
                **self._distance_params,
            )
            self.update_distance_calls += num_distance_calls
            cluster_centres[j] = curr_centre
            distances_to_centres[labels == j] = dist_to_centre

        return cluster_centres, distances_to_centres

    def _handle_empty_cluster(
        self,
        X: np.ndarray,
        cluster_centres: np.ndarray,
        distances_to_centres: np.ndarray,
        labels: np.ndarray,
    ):
        empty_clusters = np.setdiff1d(np.arange(self.n_clusters), labels)
        j = 0
        if empty_clusters.size > 0:
            print("Handling empty cluster")

        while empty_clusters.size > 0:
            current_empty_cluster_index = empty_clusters[0]
            index_furthest_from_centre = distances_to_centres.argmax()
            cluster_centres[current_empty_cluster_index] = X[index_furthest_from_centre]
            curr_pw = pairwise_distance(
                X, cluster_centres, metric=self.distance, **self._distance_params
            )
            self.empty_cluster_distance_calls += len(X) * len(cluster_centres)
            labels = curr_pw.argmin(axis=1)
            distances_to_centres = curr_pw.min(axis=1)
            empty_clusters = np.setdiff1d(np.arange(self.n_clusters), labels)
            j += 1
            if j > self.n_clusters:
                raise EmptyClusterError

        return labels, cluster_centres, distances_to_centres

    def _elastic_kmeans_plus_plus(
        self,
        X,
    ):
        initial_center_idx = self._random_state.randint(X.shape[0])
        indexes = [initial_center_idx]

        min_distances = pairwise_distance(
            X, X[initial_center_idx], metric=self.distance, **self._distance_params
        ).flatten()
        self.init_distance_calls += len(X) * len(X[initial_center_idx])
        labels = np.zeros(X.shape[0], dtype=int)

        for i in range(1, self.n_clusters):
            probabilities = min_distances / min_distances.sum()
            next_center_idx = self._random_state.choice(X.shape[0], p=probabilities)
            indexes.append(next_center_idx)

            new_distances = pairwise_distance(
                X, X[next_center_idx], metric=self.distance, **self._distance_params
            ).flatten()
            self.init_distance_calls += len(X) * len(X[next_center_idx])

            closer_points = new_distances < min_distances
            min_distances[closer_points] = new_distances[closer_points]
            labels[closer_points] = i

        centers = X[indexes]
        return centers, min_distances, labels

    def _check_params(self, X: np.ndarray) -> None:
        self._random_state = check_random_state(self.random_state)

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than "
                f"n_cases ({X.shape[0]})"
            )

        self._distance_params = {
            "window": self.window,
            **(self.distance_params or {}),
        }
