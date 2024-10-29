"""Time series kmeans."""

from typing import Optional, Tuple

__maintainer__ = []

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering.averaging import (
    VALID_BA_METRICS,
    elastic_barycenter_average,
    random_subset_ssg_barycenter_average,
)
from aeon.clustering.averaging._averaging import _resolve_average_callable
from aeon.clustering.base import BaseClusterer
from aeon.clustering.tests.test_k_shape import inertia

# from aeon.distances import distance, pairwise_distance
from aeon.distances import distance as temp_distance
from aeon.distances import pairwise_distance as temp_pairwise_distance


class EmptyClusterError(Exception):
    """Error raised when an empty cluster is encountered."""

    pass


class KESBA(BaseClusterer):
    """KESBA clustering implementation.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    init : str or np.ndarray, default='random'
        Random is the default and simply chooses k time series at random as
        centroids. It is fast but sometimes yields sub-optimal clustering.
        Kmeans++ [2] and is slower but often more
        accurate than random. It works by choosing centroids that are distant
        from one another.
        First is the fastest method and simply chooses the first k time series as
        centroids.
        If a np.ndarray provided it must be of shape (n_clusters, n_channels,
        n_timepoints)
        and contains the time series to use as centroids.
    distance : str or Callable, default='msm'
        Distance metric to compute similarity between time series. A list of valid
        strings for metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function`. If a callable is passed it must be
        a function that takes two 2d numpy arrays as input and returns a float.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol : float, default=1e-6
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose : bool, default=False
        Verbosity mode.
    random_state : int, np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
        If `int`, random_state is the seed used by the random number generator;
        If `np.random.RandomState` instance,
        random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    distance_params : dict, default=None
        Dictionary containing kwargs for the distance being used. For example if you
        wanted to specify a window for DTW you would pass
        distance_params={"window": 0.2}. See documentation of aeon.distances for more
        details.
    algorithm: str, default='lloyds'
        The kmeans algorithm to use. Valid strings are ["lloyds', "elkan"]. Elkan is
        faster but requires the distance measure used to satisfy the triangle
        inequality. Lloyds is slower but can work with any distance measure.

    Attributes
    ----------
    cluster_centers_ : 3d np.ndarray
        Array of shape (n_clusters, n_channels, n_timepoints))
        Time series that represent each of the cluster centers.
    labels_ : 1d np.ndarray
        1d array of shape (n_case,)
        Labels that is the index each time series belongs to.
    inertia_ : float
        Sum of distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_ : int
        Number of iterations run.

    References
    ----------
    .. [1] Holder, Christopher & Middlehurst, Matthew & Bagnall, Anthony. (2022).
    A Review and Evaluation of Elastic Distance Functions for Time Series Clustering.
    10.48550/arXiv.2205.15181.

    .. [2] Arthur, David & Vassilvitskii, Sergei. (2007). K-Means++: The Advantages of
    Careful Seeding. Proc. of the Annu. ACM-SIAM Symp. on Discrete Algorithms.
    8. 1027-1035. 10.1145/1283383.1283494.

    .. [3] Holder, Christopher & Guijo-Rubio, David & Bagnall, Anthony. (2023).
    Clustering time series with k-medoids based algorithms.
    In proceedings of the 8th Workshop on Advanced Analytics and Learning on Temporal
    Data (AALTD 2023).

    .. [4] Ali Ismail-Fawaz & Hassan Ismail Fawaz & Francois Petitjean &
    Maxime Devanne & Jonathan Weber & Stefano Berretti & Geoffrey I. Webb &
    Germain Forestier ShapeDBA: Generating Effective Time Series
    Prototypes using ShapeDTW Barycenter Averaging.
    In proceedings of the 8th Workshop on Advanced Analytics and Learning on Temporal
    Data (AALTD 2023).

    ..[5] Lloyd, S. P. (1982). Least squares quantization in pcm. IEEE Trans. Inf.
    Theory, 28:129–136.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.clustering import TimeSeriesKMeans
    >>> X = np.random.random(size=(10,2,20))
    >>> clst= TimeSeriesKMeans(distance="euclidean",n_clusters=2)
    >>> clst.fit(X)
    TimeSeriesKMeans(distance='euclidean', n_clusters=2)
    >>> preds = clst.predict(X)
    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        # init: Union[str, np.ndarray] = "random",
        distance: Union[str, Callable] = "msm",
        ba_subset_size: float = 0.5,
        initial_step_size: float = 0.05,
        final_step_size: float = 0.005,
        window: float = 0.5,
        # n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
        distance_params: Optional[dict] = None,
    ):
        self.distance = distance
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.num_distance_calls = 0
        self.initial_step_size = initial_step_size
        self.final_step_size = final_step_size

        self.window = window
        self.ba_subset_size = ba_subset_size

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None

        super().__init__(n_clusters)

    def _fit(self, X: np.ndarray, y=None):
        self._check_params(X)

        # Initialize the cluster centres
        # cluster_centres, distances_to_centres, labels = (
        #     self._kmeans_plus_plus_center_initializer(X)
        # )
        cluster_centres, distances_to_centres, labels = self._first(X)

        inertia = np.inf
        prev_inertia = np.inf
        prev_labels = np.zeros(len(X), dtype=int)
        iters = 0

        for iters in range(self.max_iter):

            # for j in range(self.n_clusters):
            #     # cluster_centres[j], dist_to_centre = random_subset_ssg_barycenter_average(
            #     #     X[labels == j],
            #     #     distance=self.distance,
            #     #     max_iters=self.max_iter,
            #     #     tol=self.tol,
            #     #     init_barycenter="mean",
            #     #     initial_step_size=self.initial_step_size,
            #     #     final_step_size=self.final_step_size,
            #     #     verbose=self.verbose,
            #     #     random_state=self._random_state,
            #     #     ba_subset_size=self.ba_subset_size,
            #     #     return_distances=True,
            #     #     **self._distance_params,
            #     # )
            #     cluster_centres[j], dist_to_centre = elastic_barycenter_average(
            #         X[labels == j],
            #         distance=self.distance,
            #         # max_iters=self.max_iter,
            #         # tol=self.tol,
            #         # init_barycenter="mean",
            #         random_state=self._random_state,
            #         return_distances=True,
            #         **self._distance_params,
            #     )
            #     distances_to_centres[labels == j] = dist_to_centre
            #
            # # Additional verbose output
            # if self.verbose:
            #     print(f"Iteration {iters}, inertia {inertia:.3f}.")

            distance_between_centres = self.pairwise_distance(
                cluster_centres,
                cluster_centres,
                metric=self.distance,
                **self._distance_params,
            )

            labels, distances_to_centres, inertia = self._kesba_assignment(
                X,
                cluster_centres,
                distances_to_centres,
                labels,
                iters == 0,
                distance_between_centres,
            )

            if np.array_equal(labels, prev_labels):
                if self.verbose:
                    print(f"Converged at iteration {iters}, inertia {inertia:.5f}.")

                if prev_inertia < inertia:
                    inertia = prev_inertia
                    labels = prev_labels
                break

            prev_inertia = inertia
            prev_labels = labels.copy()
            for j in range(self.n_clusters):
                # cluster_centres[j], dist_to_centre = random_subset_ssg_barycenter_average(
                #     X[labels == j],
                #     distance=self.distance,
                #     max_iters=self.max_iter,
                #     tol=self.tol,
                #     init_barycenter="mean",
                #     initial_step_size=self.initial_step_size,
                #     final_step_size=self.final_step_size,
                #     verbose=self.verbose,
                #     random_state=self._random_state,
                #     ba_subset_size=self.ba_subset_size,
                #     return_distances=True,
                #     **self._distance_params,
                # )
                cluster_centres[j], dist_to_centre = elastic_barycenter_average(
                    X[labels == j],
                    distance=self.distance,
                    # max_iters=self.max_iter,
                    # tol=self.tol,
                    # init_barycenter="mean",
                    random_state=self._random_state,
                    return_distances=True,
                    **self._distance_params,
                )
                distances_to_centres[labels == j] = dist_to_centre

            # Additional verbose output
            if self.verbose:
                print(f"Iteration {iters}, inertia {inertia:.3f}.")

        if self.verbose:
            print(f"Number of distance calls: {self.num_distance_calls}")

        self.labels_ = labels
        self.inertia_ = inertia
        self.cluster_centers_ = cluster_centres
        self.n_iter_ = iters + 1

    def _check_stopping_condition(self, prev_inertia, inertia):
        change_in_inertia = np.abs(prev_inertia - inertia)
        if change_in_inertia < self.tol:
            if self.verbose:
                print(  # noqa: T001
                    f"Converged at iteration {self.n_iter_}, inertia {inertia:.5f}."
                )
            return True
        return False

    def _handle_empty_cluster(
        self,
        X: np.ndarray,
        cluster_centres: np.ndarray,
        distances_to_centres: np.ndarray,
        labels: np.ndarray,
    ):
        empty_clusters = np.setdiff1d(np.arange(self.n_clusters), labels)
        j = 0

        while empty_clusters.size > 0:
            # Assign each time series to the cluster that is closest to it
            # and then find the time series that is furthest from its assigned centre
            current_empty_cluster_index = empty_clusters[0]
            index_furthest_from_centre = distances_to_centres.argmax()
            cluster_centres[current_empty_cluster_index] = X[index_furthest_from_centre]
            curr_pw = self.pairwise_distance(
                X, cluster_centres, metric=self.distance, **self._distance_params
            )
            labels = curr_pw.argmin(axis=1)
            distances_to_centres = curr_pw.min(axis=1)
            empty_clusters = np.setdiff1d(np.arange(self.n_clusters), labels)
            j += 1
            if j > self.n_clusters:
                # This should be unreachable but just a safety check to stop it looping
                # forever
                raise EmptyClusterError

        return labels, cluster_centres, distances_to_centres

    def _kesba_assignment(
        self,
        X,
        cluster_centres,
        distances_to_centres,
        labels,
        is_first_iteration,
        distances_between_centres,
    ):
        for i in range(X.shape[0]):
            min_dist = distances_to_centres[i]
            closest = labels[i]
            for j in range(self.n_clusters):
                if not is_first_iteration and j == closest:
                    continue
                bound = distances_between_centres[j, closest] / 2.0
                if min_dist < bound:
                    continue

                dist = self.distance_comp(
                    X[i],
                    cluster_centres[j],
                    metric=self.distance,
                    **self._distance_params,
                )
                if dist < min_dist:
                    min_dist = dist
                    closest = j

            labels[i] = closest
            distances_to_centres[i] = min_dist

        inertia = np.sum(distances_to_centres**2)
        if self.verbose:
            print(f"{inertia:.3f}", end=" --> ")
        return labels, distances_to_centres, inertia

    def pairwise_distance(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        metric=None,
        **kwargs,
    ) -> np.ndarray:
        if y is not None:
            self.num_distance_calls += x.shape[0] * y.shape[0]
        else:
            self.num_distance_calls += x.shape[0] * x.shape[0]

        return temp_pairwise_distance(
            x,
            y,
            metric=self.distance,
            **self._distance_params,
        )

    def distance_comp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metric=None,
        **kwargs,
    ) -> float:
        self.num_distance_calls += 1

        return temp_distance(
            x,
            y,
            metric=self.distance,
            **self._distance_params,
        )

    def _score(self, X, y=None):
        return -self.inertia_

    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        if isinstance(self.distance, str):
            pairwise_matrix = self.pairwise_distance(
                X, self.cluster_centers_, metric=self.distance, **self._distance_params
            )
        else:
            pairwise_matrix = self.pairwise_distance(
                X,
                self.cluster_centers_,
                metric=self.distance,
                **self._distance_params,
            )
        return pairwise_matrix.argmin(axis=1)

    def _check_params(self, X: np.ndarray) -> None:
        self._random_state = check_random_state(self.random_state)
        self.num_distance_calls = 0

        if self.distance_params is None:
            self._distance_params = {}
        else:
            self._distance_params = self.distance_params

        self._distance_params["window"] = self.window

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than "
                f"n_cases ({X.shape[0]})"
            )

    def _kmeans_plus_plus_center_initializer(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        initial_center_idx = self._random_state.randint(X.shape[0])
        indexes = [initial_center_idx]

        min_distances = self.pairwise_distance(
            X, X[initial_center_idx], metric=self.distance, **self._distance_params
        ).flatten()
        labels = np.zeros(X.shape[0], dtype=int)
        indexes.append(initial_center_idx)

        for i in range(1, self.n_clusters):
            probabilities = min_distances / min_distances.sum()
            next_center_idx = self._random_state.choice(X.shape[0], p=probabilities)
            indexes.append(next_center_idx)

            new_distances = self.pairwise_distance(
                X, X[next_center_idx], metric=self.distance, **self._distance_params
            ).flatten()

            closer_points = new_distances < min_distances
            min_distances[closer_points] = new_distances[closer_points]
            labels[closer_points] = (
                i  # Assign label i to points closer to the new center
            )

            # min_distances = np.minimum(min_distances, new_distances)

        centers = X[indexes]
        return centers, min_distances, labels

    def _first(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        first = X[: self.n_clusters]
        pw = self.pairwise_distance(
            X, first, metric=self.distance, **self._distance_params
        )
        labels = pw.argmin(axis=1)
        distances_to_centres = pw.min(axis=1)
        return first, distances_to_centres, labels

    # def _kmeans_plus_plus_center_initializer(self, X: np.ndarray):
    #     initial_center_idx = self._random_state.randint(X.shape[0])
    #     indexes = [initial_center_idx]
    #
    #     for i in range(1, self.n_clusters):
    #         pw_dist = self.pairwise_distance(
    #             X, X[indexes], metric=self.distance, **self._distance_params
    #         )
    #         min_distances = pw_dist.min(axis=1)
    #         probabilities = min_distances / min_distances.sum()
    #         next_center_idx = self._random_state.choice(X.shape[0], p=probabilities)
    #         indexes.append(next_center_idx)
    #
    #     centers = X[indexes]
    #     return centers


# if np.unique(labels).size < self.n_clusters:
#     # Recompute distances and labels to handle empty clusters
#     curr_pw = self.pairwise_distance(
#         X, cluster_centres, metric=self.distance, **self._distance_params
#     )
#     labels = curr_pw.argmin(axis=1)
#
#     # Handle empty clusters
#     labels, cluster_centres, distances_to_centres = (
#         self._handle_empty_cluster(
#             X, cluster_centres, curr_pw, labels
#         )
#     )
#     inertia = np.sum(distances_to_centres ** 2)

# if self._check_stopping_condition(prev_inertia, inertia):
#     if prev_inertia < inertia:
#         inertia = prev_inertia
#         labels = prev_labels
#     break
