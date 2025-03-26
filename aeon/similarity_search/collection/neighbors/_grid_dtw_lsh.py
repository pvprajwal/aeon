"""."""

import numpy as np
from numba import get_num_threads, njit, set_num_threads

from aeon.similarity_search.collection._base import BaseCollectionSimilaritySearch


@njit(cache=True, fastmath=True)
def _series_to_grid_coord(X, delta, normalize):
    """Series to grid coordinates."""
    n_channels, n_timepoints = X.shape
    X_hash = ""
    for i in range(n_channels):
        seq = np.zeros(n_timepoints, dtype=np.int64)
        for j in range(0, n_timepoints):
            seq[j] = np.int64(X[i, j] // delta)
        if normalize:
            seq -= np.min(seq)
        X_hash += "|"
        last_hash = ""
        for j in range(0, n_timepoints):
            x_val = str(seq[j])
            if x_val != last_hash:
                X_hash += x_val
                last_hash = x_val
    return X_hash


@njit(cache=True)
def _hash_series(X, delta, T, K, normalize):
    n_ts = X.shape[1]
    step = np.int64(n_ts // K)
    rems = np.zeros(K, dtype=np.int64)
    rems[-1] = n_ts % K
    return "*".join(
        [
            _series_to_grid_coord(
                X[:, i * step : rems[i] + (i + 1) * step], delta + T[i], normalize
            )
            for i in range(K)
        ]
    )


def _collection_index_dict(
    X,
    deltas,
    hashes_grid_shift,
    L,
    K,
    n_jobs,
    normalize,
    stride=0.2,
):
    # n_hashes = hashes_grid_shift.shape[0]
    n_hashes = len(deltas)
    index = {}
    for i in range(len(X)):
        for length in L:
            _stride = max(2, int(length * stride))
            for j in range(0, X[i].shape[1] - length + 1, _stride):
                sub = X[i][:, j : j + length]
                key = "$".join(
                    [
                        _hash_series(
                            sub,
                            deltas[i_hash],
                            hashes_grid_shift[i_hash],
                            K,
                            normalize,
                        )
                        for i_hash in range(n_hashes)
                    ]
                )
                if key not in index:
                    index[key] = []
                index[key].append([i, j, length])
    for key in index:
        index[key] = np.asarray(index[key])
    return index


class GridIndexANN(BaseCollectionSimilaritySearch):
    """
    Locality Sensitive Hashing index for Freshet and DTW distance as described in [1]_.

    Note that this method will not provide exact results, but will perform approximate
    searches.

    Examples
    --------
    from aeon.datasets import load_classification

    X_train, y_train = load_classification("GunPoint", split="train")
    X_test, y_test = load_classification("GunPoint", split="test")
    g = GridIndexANN(
        0.33,
        3,
        n_hash_funcs=2,
        normalize=True,
        n_jobs=-1,
        random_state=42,
    ).fit(X_train, y_train)
    idx, dists = g.predict(X_test, k=3)


    References
    ----------
    .. [1] Anne Driemel and Francesco Silvestri. Locality-Sensitive Hashing of Curves.
    In 33rd International Symposium on Computational Geometry (SoCG 2017).
    """

    _tags = {
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "X_inner_type": ["numpy3D", "np-list"],
    }

    def __init__(
        self,
        grid_deltas,
        K,
        L_max=1.0,
        L_min=9,
        L_step=0.05,
        random_state=None,
        normalize=True,
        n_jobs=1,
    ):
        self.grid_deltas = np.array(grid_deltas, dtype=float)
        self.K = K
        self.L_min = L_min
        self.L_max = L_max
        self.L_step = L_step
        self.random_state = random_state
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.n_hash_funcs = len(grid_deltas)
        super().__init__()

    def _fit(self, X, y=None):
        """
        Build the index based on the X.

        Parameters
        ----------
        X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
            Input array to be used to build the index.
        y : optional
            Not used.

        Returns
        -------
        self

        """
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        rng = np.random.default_rng(self.random_state)
        if self.min_n_timestamps_ < self.K:
            raise ValueError(
                f"GridIndexANN only supports time series with more than K ({self.K}) "
                f"timepoints but got a series with {self.min_n_timestamps_} timepoint"
                "in X."
            )
        self.L = np.arange(
            max(self.K, self.L_min),
            int(self.min_n_timestamps_ * self.L_max),
            max(5, int(self.L_step * self.min_n_timestamps_)),
        )

        _u_shit = self.grid_deltas / 4
        self.hashes_grid_shift = np.zeros((self.n_hash_funcs, self.K))
        for i in range(self.n_hash_funcs):
            self.hashes_grid_shift[i] = rng.uniform(
                low=max(self.grid_deltas[i] - _u_shit[i], 0),
                high=self.grid_deltas[i] + _u_shit[i],
                size=self.K,
            )

        self.index_ = _collection_index_dict(
            X,
            self.grid_deltas,
            self.hashes_grid_shift,
            self.L,
            self.K,
            self._n_jobs,
            self.normalize,
        )
        set_num_threads(prev_threads)
        return self

    def _predict(
        self,
        X,
        k=1,
        inverse_distance=False,
    ):
        """
        Find approximate nearest neighbors of a collection in the index.

        Parameters
        ----------
        X : np.ndarray, shape = (n_cases, n_channels, n_tiempoints)
            Collections of series for which we want to find neighbors.
        k : int, optional
            Number of neighbors to return for each series. The default is 1.
        inverse_distance : bool, optional
            Wheter to inverse the computed distance, meaning that the method will return
            the k most dissimilar neighbors instead of the k most similar.

        Returns
        -------
        top_k : np.ndarray, shape = (n_cases, k)
            Indexes of k series in the index that are similar to X.
        top_k_dist : np.ndarray, shape = (n_cases, k)
            Distance of k series in the index to X. The distance
            is the hamming distance between the result of each hash function.
        """
        # add check of min/max timestmaps
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        pred_index = _collection_index_dict(
            X,
            self.grid_deltas,
            self.hashes_grid_shift,
            self.L,
            self.K,
            self._n_jobs,
            self.normalize,
        )

        sims = np.zeros((len(X), self.n_cases_))
        top_k = np.zeros((len(X), k), dtype=int)
        top_k_dist = np.zeros((len(X), k))
        for key, bucket in self.index_.items():
            if key in pred_index:
                i_preds, counts_pred = np.unique(
                    pred_index[key][:, 0], return_counts=True
                )
                i_train, counts_train = np.unique(bucket[:, 0], return_counts=True)
                for i in range(len(i_preds)):
                    # N subs of one X_pred in bucket - N subs of all X_train in bucket
                    sims[i_preds[i], i_train] += 1 / (
                        1 + np.abs(counts_pred[i] - counts_train)
                    )
        sims /= len(self.index_)
        for i in range(len(X)):
            if inverse_distance:
                argmax = np.argsort(sims[i])[:k]
            else:
                argmax = np.argsort(sims[i])[::-1][:k]
            top_k[i] = argmax
            top_k_dist[i] = sims[i, argmax]
        set_num_threads(prev_threads)
        return top_k, top_k_dist
