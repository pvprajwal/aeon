"""Time series kmeans."""

from typing import Optional

__maintainer__ = []

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from aeon.clustering.averaging import VALID_BA_METRICS
from aeon.clustering.averaging._averaging import _resolve_average_callable
from aeon.clustering.base import BaseClusterer

# from aeon.distances import distance, pairwise_distance
from aeon.distances import distance as temp_distance
from aeon.distances import pairwise_distance as temp_pairwise_distance


class EmptyClusterError(Exception):
    """Error raised when an empty cluster is encountered."""

    pass


class TimeSeriesKMeans(BaseClusterer):
    """Time series K-means clustering implementation.

    K-means [5]_ is a popular clustering algorithm that aims to partition n time series
    into k clusters in which each observation belongs to the cluster with the nearest
    centre. The centre is represented using an average which is generated during the
    training phase.

    K-means using euclidean distance for time series generally performs poorly. However,
    when combined with an elastic distance it performs significantly better (in
    particular MSM/TWE [1]_). K-means for time series can further be improved by using
    an elastic averaging method. The most common one is dynamic barycenter averaging
    [3]_ however, in recent years alternates using other elastic distances such as
    ShapeDBA [4]_ (Shape DTW DBA) and MBA (Msm DBA) [5]_ have shown signicant
    performance benefits.

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
    averaging_method : str or Callable, default='ba'
        Averaging method to compute the average of a cluster. Any of the following
        strings are valid: ['mean', 'ba']. If a Callable is provided must take the form
        Callable[[np.ndarray], np.ndarray].
        If you specify 'ba' then by default the distance measure used will be the same
        as the distance measure used for clustering. If you wish to use a different
        distance measure you can specify it by passing {"distance": "dtw"} as
        averaging_params. BA yields 'better' clustering results but is very
        computationally expensive so you may want to consider setting a bounding window
        or using a different averaging method if time complexity is a concern.
    average_params : dict, default=None
        Dictionary containing kwargs for averaging_method. See documentation of
        aeon.clustering.averaging and aeon.distances for more details. NOTE: if you
        want to use custom distance params during averaging here you must specify them
        in this dict in addition to custom averaging params. For example to specify a
        window as a distance param and verbosity for the averaging you would pass
        average_params={"window": 0.2, "verbose": True}.
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
        init: Union[str, np.ndarray] = "random",
        distance: Union[str, Callable] = "msm",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
        averaging_method: Union[str, Callable[[np.ndarray], np.ndarray]] = "ba",
        distance_params: Optional[dict] = None,
        average_params: Optional[dict] = None,
        init_algorithm: Optional[Union[str, np.ndarray]] = None,
        algorithm: str = "lloyds",
    ):
        self.init = init
        self.init_algorithm = init_algorithm
        if init_algorithm is not None:
            import warnings

            warnings.warn(
                "The 'init_algorithm' parameter is deprecated and will be "
                "removed in a future. Version Use 'init' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.init = self.init_algorithm
        self.distance = distance
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.average_params = average_params
        self.averaging_method = averaging_method
        self.algorithm = algorithm
        self.num_distance_calls = 0

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._init = None
        self._averaging_method = None
        self._average_params = None

        super().__init__(n_clusters)

    def _fit(self, X: np.ndarray, y=None):
        self.num_distance_calls = 0
        self._check_params(X)

        best_centers = None
        best_inertia = np.inf
        best_labels = None
        best_iters = self.max_iter

        average_iterations = 0

        for _ in range(self.n_init):
            try:
                if self.algorithm == "lloyds":
                    labels, centers, inertia, n_iters = self._fit_one_init_lloyds(X)
                elif self.algorithm == "elkan":
                    labels, centers, inertia, n_iters = self._fit_one_init_elkan(X)
                elif self.algorithm == "tony-elkan":
                    labels, centers, inertia, n_iters = self._fit_one_init_tony(X)
                else:
                    raise ValueError(
                        "Invalid algorithm specified. Must be 'lloyds' or 'elkan'"
                    )
                average_iterations += n_iters

                if inertia < best_inertia:
                    best_centers = centers
                    best_labels = labels
                    best_inertia = inertia
                    best_iters = n_iters
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")  # noqa: T001, T201

        if best_labels is None:
            raise ValueError(
                "Unable to find a valid cluster configuration "
                "with parameters specified (empty clusters kept "
                "forming). Try lowering your n_clusters or raising "
                "n_init."
            )

        if self.verbose:
            print(f"Number of distance calls: {self.num_distance_calls}")

        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers
        self.n_iter_ = best_iters

        average_iterations = average_iterations / self.n_init
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++")  # noqa E501
        print(f"Average number of iterations: {average_iterations}")  # noqa E501
        print(f"Best iteration number of iterations: {best_iters}")  # noqa E501
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")  # noqa E501

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
    ) -> np.ndarray:
        self.num_distance_calls += 1

        return temp_distance(
            x,
            y,
            metric=self.distance,
            **self._distance_params,
        )

    def _fit_one_init_tony(self, X: np.ndarray) -> tuple:
        raise NotImplementedError("Tony's elkan not implemented yet.")

    def _fit_one_init_elkan(self, X: np.ndarray) -> tuple:
        # Initialize the centroids (same as in the Lloyd's method)
        n_instances, n_channels, n_timepoints = X.shape

        if isinstance(self._init, Callable):
            cluster_centres = self._init(X)
        else:
            cluster_centres = self._init.copy()

        n_clusters = cluster_centres.shape[0]
        # Initialize labels, upper bounds, lower bounds
        curr_labels = np.zeros(n_instances, dtype=int)
        upper_bounds = np.full(n_instances, np.inf)
        lower_bounds = np.zeros((n_instances, n_clusters))

        prev_inertia = np.inf

        for i in range(self.max_iter):
            # Step 1: Compute center-center distances using pairwise_distance
            center_distances = self.pairwise_distance(
                cluster_centres,
                cluster_centres,
                metric=self.distance,
                **self._distance_params,
            )

            # Step 2: Compute half the minimum distance to other centers for each center
            center_half_min_dist = 0.5 * np.min(
                center_distances + np.diag([np.inf] * n_clusters), axis=1
            )

            # Step 3: Assignment step with bounds
            for idx in range(n_instances):
                current_center = curr_labels[idx]
                if upper_bounds[idx] <= center_half_min_dist[current_center]:
                    continue  # Current center is sufficiently close
                needs_update = True
                for j in range(n_clusters):
                    if j == current_center:
                        continue
                    z = max(
                        lower_bounds[idx, j], 0.5 * center_distances[current_center, j]
                    )
                    if upper_bounds[idx] <= z:
                        continue  # No closer center possible
                    if needs_update:
                        # Compute the exact distance to the assigned center
                        upper_bounds[idx] = self.distance_comp(
                            X[idx],
                            cluster_centres[current_center],
                            metric=self.distance,
                            **self._distance_params,
                        )
                        needs_update = False
                        if upper_bounds[idx] <= z:
                            continue  # No closer center possible
                    # Compute distance to center j
                    lower_bounds[idx, j] = self.distance_comp(
                        X[idx],
                        cluster_centres[j],
                        metric=self.distance,
                        **self._distance_params,
                    )
                    if lower_bounds[idx, j] < upper_bounds[idx]:
                        curr_labels[idx] = j
                        upper_bounds[idx] = lower_bounds[idx, j]
                        current_center = j  # Update current_center

            # Compute current inertia using upper bounds (squared distances)
            curr_inertia = np.sum(upper_bounds**2)

            # Check for empty clusters
            if np.unique(curr_labels).size < self.n_clusters:
                # Recompute distances and labels to handle empty clusters
                curr_pw = self.pairwise_distance(
                    X, cluster_centres, metric=self.distance, **self._distance_params
                )
                curr_labels = curr_pw.argmin(axis=1)
                curr_inertia = curr_pw.min(axis=1).sum()

                # Handle empty clusters
                curr_pw, curr_labels, curr_inertia, cluster_centres = (
                    self._handle_empty_cluster(
                        X, cluster_centres, curr_pw, curr_labels, curr_inertia
                    )
                )

                # Update upper_bounds based on new assignments
                upper_bounds = curr_pw[np.arange(n_instances), curr_labels]
                # Reset lower bounds
                lower_bounds = np.zeros((n_instances, n_clusters))

            # Verbose output
            if self.verbose:
                print(f"{curr_inertia:.3f}", end=" --> ")

            # Check for convergence based on change in inertia
            change_in_inertia = np.abs(prev_inertia - curr_inertia)
            if change_in_inertia < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}, inertia {curr_inertia:.3f}.")
                break

            prev_inertia = curr_inertia

            # Step 4: Update the centers using the averaging method
            new_cluster_centres = np.zeros_like(cluster_centres)
            for j in range(n_clusters):
                assigned_points = X[curr_labels == j]
                new_cluster_centres[j] = self._averaging_method(
                    assigned_points, **self._average_params
                )

            # Step 5: Compute center shift distances
            center_shifts = np.array(
                [
                    self.distance_comp(
                        cluster_centres[j],
                        new_cluster_centres[j],
                        metric=self.distance,
                        **self._distance_params,
                    )
                    for j in range(n_clusters)
                ]
            )

            cluster_centres = new_cluster_centres

            # Step 6: Update upper and lower bounds
            upper_bounds += center_shifts[curr_labels]
            for j in range(n_clusters):
                lower_bounds[:, j] = np.maximum(
                    lower_bounds[:, j] - center_shifts[j], 0
                )

            # Additional verbose output
            if self.verbose:
                print(f"Iteration {i}, inertia {curr_inertia:.3f}.")

        else:
            if self.verbose:
                print(
                    f"Reached maximum iterations {self.max_iter}, inertia {curr_inertia:.5f}."
                )

        return curr_labels, cluster_centres, curr_inertia, i + 1

    def _fit_one_init_lloyds(self, X: np.ndarray) -> tuple:
        if isinstance(self._init, Callable):
            cluster_centres = self._init(X)
        else:
            cluster_centres = self._init.copy()
        prev_inertia = np.inf
        prev_labels = None
        for i in range(self.max_iter):
            curr_pw = self.pairwise_distance(
                X, cluster_centres, metric=self.distance, **self._distance_params
            )
            curr_labels = curr_pw.argmin(axis=1)
            curr_inertia = curr_pw.min(axis=1).sum()

            # If an empty cluster is encountered
            if np.unique(curr_labels).size < self.n_clusters:
                curr_pw, curr_labels, curr_inertia, cluster_centres = (
                    self._handle_empty_cluster(
                        X, cluster_centres, curr_pw, curr_labels, curr_inertia
                    )
                )

            if self.verbose:
                print("%.3f" % curr_inertia, end=" --> ")  # noqa: T001, T201

            change_in_centres = np.abs(prev_inertia - curr_inertia)
            prev_inertia = curr_inertia
            prev_labels = curr_labels

            if change_in_centres < self.tol:
                print(  # noqa: T001
                    f"Converged at iteration {i}, inertia {curr_inertia:.5f}."
                )
                break

            # Compute new cluster centres
            num_iterations_ssg = []
            for j in range(self.n_clusters):
                new_average = self._averaging_method(
                    X[curr_labels == j], **self._average_params
                )
                if isinstance(new_average, tuple):
                    cluster_centres[j] = new_average[0]
                    num_iterations_ssg.append(new_average[1])
                else:
                    cluster_centres[j] = new_average

            if self.verbose is True:
                print(  # noqa: T201
                    f"++++++++++++++ Iteration {i} ++++++++++++++"  # noqa: T201
                )  # noqa: T201
                if len(num_iterations_ssg) > 0:
                    print(  # noqa: T201
                        f"Number of iterations for each "  # noqa: T201
                        f"cluster: {num_iterations_ssg}"  # noqa: T201
                    )  # noqa: T201

        return prev_labels, cluster_centres, prev_inertia, i + 1

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

        if isinstance(self.init, str):
            if self.init == "random":
                self._init = self._random_center_initializer
            elif self.init == "kmeans++":
                self._init = self._kmeans_plus_plus_center_initializer
            elif self.init == "first":
                self._init = self._first_center_initializer
        else:
            if isinstance(self.init, np.ndarray) and len(self.init) == self.n_clusters:
                self._init = self.init.copy()
            else:
                raise ValueError(
                    f"The value provided for init: {self.init} is "
                    f"invalid. The following are a list of valid init algorithms "
                    f"strings: random, kmedoids++, first. You can also pass a"
                    f"np.ndarray of size (n_clusters, n_channels, n_timepoints)"
                )

        if self.distance_params is None:
            self._distance_params = {}
        else:
            self._distance_params = self.distance_params
        if self.average_params is None:
            self._average_params = {}
        else:
            self._average_params = self.average_params

        # Add the distance to average params
        if "distance" not in self._average_params:
            # Must be a str and a valid distance for ba averaging
            if isinstance(self.distance, str) and self.distance in VALID_BA_METRICS:
                self._average_params["distance"] = self.distance
            else:
                # Invalid distance passed for ba so default to dba
                self._average_params["distance"] = "dtw"

        if "random_state" not in self._average_params:
            self._average_params["random_state"] = self._random_state

        self._averaging_method = _resolve_average_callable(self.averaging_method)

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be larger than "
                f"n_cases ({X.shape[0]})"
            )

    def _random_center_initializer(self, X: np.ndarray) -> np.ndarray:
        return X[self._random_state.choice(X.shape[0], self.n_clusters, replace=False)]

    def _first_center_initializer(self, X: np.ndarray) -> np.ndarray:
        return X[list(range(self.n_clusters))]

    def _kmeans_plus_plus_center_initializer(self, X: np.ndarray):
        initial_center_idx = self._random_state.randint(X.shape[0])
        indexes = [initial_center_idx]

        for _ in range(1, self.n_clusters):
            pw_dist = self.pairwise_distance(
                X, X[indexes], metric=self.distance, **self._distance_params
            )
            min_distances = pw_dist.min(axis=1)
            probabilities = np.abs(min_distances / min_distances.sum())
            next_center_idx = self._random_state.choice(X.shape[0], p=probabilities)
            indexes.append(next_center_idx)

        centers = X[indexes]
        return centers

    def _random_old_center_initializer(self, X: np.ndarray) -> np.ndarray:
        # assign each time series to a random cluster
        max_attempts = 200
        attempts = 0
        while True:
            labels = self._random_state.choice(self.n_clusters, size=X.shape[0])
            # Check one value in every cluster
            temp = set(labels)
            if len(temp) == self.n_clusters:
                break
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    "Failed to initialize cluster centers after 200 attempts"
                )

        cluster_centers = np.zeros((self.n_clusters, X.shape[1], X.shape[2]))

        for i in range(self.n_clusters):
            cluster_centers[i] = self._averaging_method(
                X[labels == i], **self._average_params
            )

        return cluster_centers

    def _handle_empty_cluster(
        self,
        X: np.ndarray,
        cluster_centres: np.ndarray,
        curr_pw: np.ndarray,
        curr_labels: np.ndarray,
        curr_inertia: float,
    ):
        """Handle an empty cluster.

        This functions finds the time series that is furthest from its assigned centre
        and then uses that as the new centre for the empty cluster. In terms of
        optimisation this means it selects the time series that will reduce inertia
        by the most.
        """
        empty_clusters = np.setdiff1d(np.arange(self.n_clusters), curr_labels)
        j = 0

        while empty_clusters.size > 0:
            # Assign each time series to the cluster that is closest to it
            # and then find the time series that is furthest from its assigned centre
            current_empty_cluster_index = empty_clusters[0]
            index_furthest_from_centre = curr_pw.min(axis=1).argmax()
            cluster_centres[current_empty_cluster_index] = X[index_furthest_from_centre]
            curr_pw = self.pairwise_distance(
                X, cluster_centres, metric=self.distance, **self._distance_params
            )
            curr_labels = curr_pw.argmin(axis=1)
            curr_inertia = curr_pw.min(axis=1).sum()
            empty_clusters = np.setdiff1d(np.arange(self.n_clusters), curr_labels)
            j += 1
            if j > self.n_clusters:
                # This should be unreachable but just a safety check to stop it looping
                # forever
                raise EmptyClusterError
        return curr_pw, curr_labels, curr_inertia, cluster_centres

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {
            "n_clusters": 2,
            "distance": "euclidean",
            "n_init": 1,
            "max_iter": 1,
            "random_state": 0,
            "averaging_method": "mean",
        }
