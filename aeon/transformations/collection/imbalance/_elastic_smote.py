from collections import OrderedDict
from typing import Optional, Union

import numpy as np
from numba import njit, prange
from sklearn.utils import check_random_state

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.clustering.averaging._ba_utils import _get_alignment_path
from aeon.transformations.collection import BaseCollectionTransformer

__maintainer__ = ["chrisholder"]
__all__ = ["ElasticSMOTE"]

from aeon.utils._threading import threaded


class ElasticSMOTE(BaseCollectionTransformer):
    """
    Over-sampling using the Synthetic Minority Over-sampling TEchnique (SMOTE)[1]_.

    An adaptation of the imbalance-learn implementation of SMOTE in
    imblearn.over_sampling.SMOTE. sampling_strategy is sampling target by
    targeting all classes but not the majority, which is directly expressed in
    _fit.sampling_strategy.

    Parameters
    ----------
    n_neighbors : int, default=5
        The number  of nearest neighbors used to define the neighborhood of samples
        to use to generate the synthetic time series.
        `~sklearn.neighbors.NearestNeighbors` instance will be fitted in this case.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    See Also
    --------
    ADASYN

    References
    ----------
    .. [1] Chawla et al. SMOTE: synthetic minority over-sampling technique, Journal
    of Artificial Intelligence Research 16(1): 321–357, 2002.
        https://dl.acm.org/doi/10.5555/1622407.1622416
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
        self,
        n_neighbors=5,
        distance: Union[str, callable] = "euclidean",
        distance_params: Optional[dict] = None,
        weights: Union[str, callable] = "uniform",
        n_jobs: int = 1,
        random_state=None,
    ):
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.distance_params = distance_params
        self.weights = weights
        self.n_jobs = n_jobs

        self._random_state = None
        self._distance_params = distance_params or {}

        self.nn_ = None
        super().__init__()

    def _fit(self, X, y=None):
        self._random_state = check_random_state(self.random_state)
        self.nn_ = KNeighborsTimeSeriesClassifier(
            n_neighbors=self.n_neighbors + 1,
            distance=self.distance,
            distance_params=self._distance_params,
            weights=self.weights,
            n_jobs=self.n_jobs,
        )

        # generate sampling target by targeting all classes except the majority
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))
        return self

    def _transform(self, X, y=None):
        # remove the channel dimension to be compatible with sklearn
        X = np.squeeze(X, axis=1)
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # got the minority class label and the number needs to be generated
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]
            y_class = y[target_class_indices]

            self.nn_.fit(X_class, y_class)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class,
                y.dtype,
                class_sample,
                X_class,
                nns,
                n_samples,
                1.0,
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        X_resampled = X_resampled[:, np.newaxis, :]
        return X_resampled, y_resampled

    @threaded
    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0
    ):
        samples_indices = self._random_state.randint(
            low=0, high=nn_num.size, size=n_samples
        )

        steps = step_size * self._random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = _generate_samples(
            X,
            nn_data,
            nn_num,
            rows,
            cols,
            steps,
            distance=self.distance,
            **self._distance_params,
        )
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new


@njit(cache=True, fastmath=True, parallel=True)
def _generate_samples(
    X,
    nn_data,
    nn_num,
    rows,
    cols,
    steps,
    distance,
    weights: Optional[np.ndarray] = None,
    window: Union[float, None] = None,
    g: float = 0.0,
    epsilon: Union[float, None] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 15,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
):
    X_new = np.zeros((len(rows), X.shape[1]), dtype=X.dtype)

    for count in prange(len(rows)):
        i = rows[count]
        j = cols[count]
        curr_ts = X[i]
        nn_ts = nn_data[nn_num[i, j]]
        new_ts = curr_ts.copy()

        alignment, _ = _get_alignment_path(
            nn_ts,
            curr_ts,
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
        num_of_alignments = np.zeros_like(curr_ts, dtype=np.int32)
        for k, l in alignment:
            new_ts[l] += curr_ts[k] - nn_ts[l]
            num_of_alignments[l] += 1

        new_ts = new_ts / num_of_alignments
        X_new[count] = new_ts * steps[count]

    return X_new


if __name__ == "__main__":
    from aeon.testing.data_generation import make_example_3d_numpy

    X, y = make_example_3d_numpy(100, 1, 10, random_state=1)

    smote = ElasticSMOTE(n_neighbors=5, random_state=1, distance="msm")

    X_resampled, y_resampled = smote.fit_transform(X, y)
    stop = ""
