from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize

from aeon.clustering.averaging._ba_utils import _get_init_barycenter
from aeon.distances import soft_dtw_distance, squared_distance, pairwise_distance, \
    squared_pairwise_distance, soft_dtw_cost_matrix

from aeon.distances._soft_dtw import soft_dtw_gradient


def soft_barycenter_average(
        X: np.ndarray,
        distance: str = "soft_dtw",
        gamma: float = 1.0,
        minimise_method: str = "L-BFGS-B",
        max_iters: int = 30,
        tol=1e-5,
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
        distance,
        random_state=random_state,
        **kwargs,
    )
    barycenter_shape = barycenter.shape

    def f(curr_barycenter):
        curr_barycenter = curr_barycenter.reshape(barycenter_shape)
        return _softdtw_func(curr_barycenter, X, gamma=gamma, weights=weights)

    res = minimize(f, barycenter.ravel(), method=minimise_method, jac=True, tol=tol,
                   options=dict(maxiter=max_iters, disp=False))

    # X = X.swapaxes(1, 2)
    #
    # return softdtw_barycenter(
    #     X,
    #     gamma=gamma,
    #     weights=weights,
    #     method=minimise_method,
    #     tol=tol,
    #     max_iter=max_iters,
    #     init=barycenter,
    # ).swapaxes(0, 1)
def _softdtw_func(curr_barycentre, X, weights, gamma):
    G = np.zeros_like(curr_barycentre)
    obj = 0

    for i in range(len(X)):
        first_value = X[i]
        value = soft_dtw_distance(curr_barycentre, X[i], gamma=gamma)
        # Add extra row and column to distance matrix
        test = soft_dtw_gradient(curr_barycentre, X[i], gamma)

        stop = ""
    #     E = sdtw.grad()
    #     G_tmp = D.jacobian_product(E)
    #     G += weights[i] * G_tmp
    #     obj += weights[i] * value
    #
    # return obj, G.ravel()

if __name__ == "__main__":
    from aeon.testing.data_generation import make_example_3d_numpy

    X = make_example_3d_numpy(n_cases=20, n_channels=1, n_timepoints=100, return_y=False, random_state=1)

    soft_barycenter_average(X)
