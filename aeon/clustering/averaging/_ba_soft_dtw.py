from typing import Optional, Union

import numpy as np
from tslearn.barycenters import softdtw_barycenter

from aeon.clustering.averaging._ba_utils import _get_init_barycenter


def soft_dtw_barycenter_average(
    X: np.ndarray,
    gamma: float = 1.0,
    minimise_method: str = "L-BFGS-B",
    max_iters: int = 30,
    tol=1e-3,
    init_barycenter: Union[np.ndarray, str] = "mean",
    weights: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Compute the barycenter average of time series using a elastic distance.

    This implements an adapted version of 'petitjean' (original) DBA algorithm [1]_.

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
    **kwargs
        Keyword arguments to pass to the distance metric.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the average of the collection of instances provided.

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    """
    if len(X) <= 1:
        return X

    if X.ndim == 3:
        _X = X
    elif X.ndim == 2:
        _X = X.reshape((X.shape[0], 1, X.shape[1]))
    else:
        raise ValueError("X must be a 2D or 3D array")

    barycenter = _get_init_barycenter(
        _X,
        init_barycenter,
        "soft_dtw",
        random_state=random_state,
        **kwargs,
    )

    barycenter = barycenter.swapaxes(0, 1)

    X = X.swapaxes(1, 2)

    return softdtw_barycenter(
        X,
        gamma=gamma,
        weights=weights,
        method=minimise_method,
        tol=tol,
        max_iter=max_iters,
        init=barycenter,
    ).swapaxes(0, 1)
