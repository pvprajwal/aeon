from typing import Optional, Tuple, Union

import numpy as np
from numba import njit
from scipy.optimize import minimize

from aeon.clustering.averaging._ba_utils import _get_init_barycenter
from aeon.clustering.averaging._soft_ba_utils import get_minimise_method
from aeon.distances import distance
from aeon.distances._soft_dtw import _soft_dtw_cost_matrix


def soft_barycenter_average(
    X: np.ndarray,
    distance: str = "soft_dtw",
    minimise_method: str = "L-BFGS-B",
    max_iters: int = 1000,
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

    if weights is None:
        weights = np.ones(len(_X))

    barycenter = _get_init_barycenter(
        _X,
        init_barycenter,
        distance,
        random_state=random_state,
        **kwargs,
    )
    barycenter_shape = barycenter.shape
    minimise_function = get_minimise_method(_X, barycenter, weights, distance, **kwargs)

    res = minimize(
        minimise_function,
        barycenter.ravel(),
        method=minimise_method,
        jac=True,
        tol=tol,
        options=dict(maxiter=max_iters, disp=True),
    )
    return res.x.reshape(barycenter_shape)


# if __name__ == "__main__":
#     from aeon.clustering.averaging import soft_dtw_barycenter_average
#     from aeon.testing.data_generation import make_example_3d_numpy
#
#     X = make_example_3d_numpy(
#         n_cases=20, n_channels=1, n_timepoints=20, return_y=False, random_state=1
#     )
#
#     test = soft_barycenter_average(X, distance="soft_msm")
#     # proper = soft_dtw_barycenter_average(X)
#     # equal = np.allclose(test, proper)
#     temp = ""
import os

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

from aeon.clustering.averaging import (
    elastic_barycenter_average,
    mean_average,
    subgradient_barycenter_average,
)
from aeon.datasets import load_from_tsfile

# PATH = "/home/chris/Documents/Univariate_ts"
PATH = "/Users/chris/Documents/Phd-data/Datasets/Univariate_ts"
dataset_name = "GunPoint"


def load_experiment_data(
    problem_path: str,
    dataset: str,
    resample_id: int,
    predefined_resample: bool,
):
    """Load data for experiments.

    Parameters
    ----------
    problem_path : str
        Path to the problem folder.
    dataset : str
        Name of the dataset.
    resample_id : int or None
        Id of the data resample to use.
    predefined_resample : boolean
        If True, use the predefined resample.

    Returns
    -------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray
        Test data labels.
    resample : boolean
        If True, the data is to be resampled.
    """
    if resample_id is not None and predefined_resample:
        resample_str = "" if resample_id is None else str(resample_id)

        X_train, y_train = load_from_tsfile(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TRAIN.ts"
        )
        X_test, y_test = load_from_tsfile(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TEST.ts"
        )

        resample_data = False
    else:
        X_train, y_train = load_from_tsfile(
            f"{problem_path}/{dataset}/{dataset}_TRAIN.ts"
        )
        X_test, y_test = load_from_tsfile(f"{problem_path}/{dataset}/{dataset}_TEST.ts")

        resample_data = True if resample_id != 0 else False

    return X_train, y_train, X_test, y_test, resample_data


import time

if __name__ == "__main__":
    # PATH = "/home/chris/Documents/Univariate_ts"
    PATH = "/home/chris/Documents/Univariate_ts"
    dataset_name = "GunPoint"
    X_train, y_train, X_test, y_test, _ = load_experiment_data(
        PATH, dataset_name, 0, False
    )
    class_one = X_train[y_train == "1"]
    data = class_one.squeeze(1)
    distance_params = {
        "window": 1.0,
    }

    # Calculate averages with squeeze adjustments
    mean = mean_average(data)

    # dba_elastic = elastic_barycenter_average(data, distance='dtw', **distance_params).squeeze(0)
    mba_elastic = elastic_barycenter_average(
        data, distance="msm", **distance_params
    ).squeeze(0)
    # soft_dtw = soft_barycenter_average(data, distance='soft_dtw', **distance_params).squeeze(0)
    soft_mba = soft_barycenter_average(
        data, distance="soft_msm", **distance_params
    ).squeeze(0)
    # soft_twe = soft_barycenter_average(data, distance='soft_twe', **distance_params).squeeze(0)
    # soft_adtw = soft_barycenter_average(data, distance='soft_adtw', **distance_params).squeeze(0)

    averages = [mean, mba_elastic, soft_mba]

    # titles = ["Mean Average", 'Holdit-DBA', 'Holdit-MBA', 'Holdit-ADTW', 'Holdit-TWE']
    titles = ["Mean Average", "MBA", "soft-MBA", "soft-DTW", "soft-dtw", "soft-ADTW"]

    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "pink",
    ]  # Define a list of colors

    print(f"Equal {np.array_equal(mean, soft_mba)}")

    plt.figure(figsize=(len(averages) * 3, 12))  # Adjust the figure size as necessary

    gs = gridspec.GridSpec(
        len(averages), 4
    )  # 5 rows for the plots, 3 columns for the new layout

    ax_original = plt.subplot(gs[2:3, 0:2])  # Centered vertically in the grid
    for i in range(data.shape[0]):
        ax_original.plot(data[i], alpha=0.5)
    ax_original.set_title("Gun point dataset class 1", fontdict={"fontsize": 20})

    positions = [0, 1, 2, 3, 4]

    for pos, (avg, color, title) in zip(positions, zip(averages, colors, titles)):
        ax = plt.subplot(gs[pos, 2:])  # Use columns 1 and 2 for these plots
        ax.plot(avg, color=color)
        ax.set_title(title, fontdict={"fontsize": 20})

    plt.tight_layout()
    plt.savefig("averaging.pdf")

    plt.show()
