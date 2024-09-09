from typing import Tuple

import numpy as np
from numba import njit

from aeon.distances import create_bounding_matrix, msm_distance, squared_distance
from aeon.distances._soft_dtw import _soft_dtw_cost_matrix

MAX_DOUBLE = np.finfo("double").max


def get_minimise_method(
    X: np.ndarray, barycenter: np.ndarray, weights: np.ndarray, distance: str, **kwargs
):
    barycenter_shape = barycenter.shape
    bounding_matrix = create_bounding_matrix(
        barycenter_shape[1],
        barycenter_shape[1],
        kwargs.get("window", None),
        kwargs.get("itakura_max_slope", None),
    )

    if distance == "soft_dtw":
        gamma = kwargs.get("gamma", 1.0)

        # @njit(cache=True, fastmath=True)
        def f(curr_barycenter):
            curr_barycenter = curr_barycenter.reshape(barycenter_shape)
            return _soft_dtw_func(curr_barycenter, X, weights, bounding_matrix, gamma)

        return f
    else:
        raise ValueError(
            "distance must be one of ['soft_dtw', 'soft_msm', 'soft_adtw', 'soft_twe']"
        )


@njit(cache=True, fastmath=True)
def _soft_dtw_func(curr_barycentre, X, weights, bounding_matrix, gamma):
    new_barycenter = np.zeros_like(curr_barycentre)
    obj = 0

    for i in range(len(X)):
        jacobian_product, distance = _soft_dtw_jacobian(
            curr_barycentre, X[i], bounding_matrix, gamma
        )
        new_barycenter += weights[i] * jacobian_product
        obj += weights[i] * distance

    return obj, new_barycenter.ravel()


@njit(cache=True, fastmath=True)
def _soft_dtw_jacobian(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, gamma: float
) -> Tuple[np.ndarray, float]:
    cost_matrix = _soft_dtw_cost_matrix(x, y, bounding_matrix, gamma)
    squared_dist = np.zeros((x.shape[1], y.shape[1]))
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            squared_dist[i, j] = squared_distance(x[:, i], y[:, j])

    gradient_matrix = compute_gradient(cost_matrix, squared_dist, gamma)
    jacobian_product = np.zeros_like(x)

    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            for channel in range(x.shape[0]):
                jacobian_product[channel, i] += (
                    gradient_matrix[i, j] * 2 * (x[channel, i] - y[channel, j])
                )
    return jacobian_product, cost_matrix[-1, -1]


@njit(cache=True, fastmath=True)
def compute_gradient(
    cost_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    gamma: float,
    max_double=MAX_DOUBLE,
) -> np.ndarray:
    m, n = distance_matrix.shape

    padded_distance_matrix = np.zeros((m + 1, n + 1))
    padded_distance_matrix[0:-1, 0:-1] = distance_matrix

    padded_cm = np.zeros((m + 2, n + 2))
    padded_cm[1:-1, 1:-1] = cost_matrix
    padded_cm[1 : m + 1, n + 1] = -max_double
    padded_cm[m + 1, 1 : n + 1] = -max_double
    padded_cm[0, : n + 1] = max_double
    padded_cm[: m + 1, 0] = max_double
    padded_cm[0, 0] = 0.0
    padded_cm[-1, -1] = cost_matrix[-1, -1]

    gradient_matrix = np.zeros((m + 2, n + 2))
    gradient_matrix[m + 1, n + 1] = 1.0

    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            a = np.exp(
                (
                    padded_cm[i + 1, j]
                    - padded_cm[i, j]
                    - padded_distance_matrix[i, j - 1]
                )
                / gamma
            )
            b = np.exp(
                (
                    padded_cm[i, j + 1]
                    - padded_cm[i, j]
                    - padded_distance_matrix[i - 1, j]
                )
                / gamma
            )
            c = np.exp(
                (
                    padded_cm[i + 1, j + 1]
                    - padded_cm[i, j]
                    - padded_distance_matrix[i, j]
                )
                / gamma
            )
            gradient_matrix[i, j] = (
                gradient_matrix[i + 1, j] * a
                + gradient_matrix[i, j + 1] * b
                + gradient_matrix[i + 1, j + 1] * c
            )

    return gradient_matrix[1:-1, 1:-1]
