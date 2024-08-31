from typing import List, Optional, Tuple, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import _univariate_squared_distance
from aeon.distances._utils import _convert_to_list, _is_multivariate, _softmin3
from aeon.distances._msm import (
    _msm_independent_cost_matrix, _msm_dependent_cost_matrix,
    _cost_independent, _cost_dependent
)


@njit(cache=True, fastmath=True)
def soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    gamma: float = 1.0,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_distance(_x, _y, bounding_matrix, independent, c, gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_distance(x, y, bounding_matrix, independent, c, gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_msm_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        if independent:
            return _soft_msm_independent_cost_matrix(_x, _y, bounding_matrix, c, gamma)
        return _soft_msm_dependent_cost_matrix(_x, _y, bounding_matrix, c, gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        if independent:
            return _soft_msm_independent_cost_matrix(x, y, bounding_matrix, c, gamma)
        return _soft_msm_dependent_cost_matrix(x, y, bounding_matrix, c, gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    independent: bool,
    c: float,
    gamma: float,
) -> float:
    if independent:
        return _soft_msm_independent_cost_matrix(x, y, bounding_matrix, c, gamma)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    return _soft_msm_dependent_cost_matrix(x, y, bounding_matrix, c, gamma)[
        x.shape[1] - 1, y.shape[1] - 1
    ]


@njit(cache=True, fastmath=True)
def _soft_msm_independent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float, gamma: float
) -> np.ndarray:
    if gamma == 0.0:
        return _msm_independent_cost_matrix(x, y, bounding_matrix, c)
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    for i in range(x.shape[0]):
        curr_cost_matrix = _soft_independent_cost_matrix(
            x[i], y[i], bounding_matrix, c, gamma
        )
        cost_matrix = np.add(cost_matrix, curr_cost_matrix)
    return cost_matrix


@njit(cache=True, fastmath=True)
def _soft_independent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float, gamma: float
) -> np.ndarray:
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = np.abs(x[0] - y[0])

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            cost = _cost_independent(x[i], x[i - 1], y[0], c)
            cost_matrix[i][0] = cost_matrix[i - 1][0] + cost

    for i in range(1, y_size):
        if bounding_matrix[0, i]:
            cost = _cost_independent(y[i], y[i - 1], x[0], c)
            cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                d1 = cost_matrix[i - 1][j - 1] + np.abs(x[i] - y[j])
                d2 = cost_matrix[i - 1][j] + _cost_independent(x[i], x[i - 1], y[j], c)
                d3 = cost_matrix[i][j - 1] + _cost_independent(y[j], x[i], y[j - 1], c)

                cost_matrix[i, j] = _softmin3(d1, d2, d3, gamma)

    return cost_matrix


@njit(cache=True, fastmath=True)
def _soft_msm_dependent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float, gamma: float
) -> np.ndarray:
    if gamma == 0.0:
        return _msm_dependent_cost_matrix(x, y, bounding_matrix, c)
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = np.sum(np.abs(x[:, 0] - y[:, 0]))

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            cost = _cost_dependent(x[:, i], x[:, i - 1], y[:, 0], c)
            cost_matrix[i][0] = cost_matrix[i - 1][0] + cost
    for i in range(1, y_size):
        if bounding_matrix[0, i]:
            cost = _cost_dependent(y[:, i], y[:, i - 1], x[:, 0], c)
            cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                d1 = cost_matrix[i - 1][j - 1] + np.sum(np.abs(x[:, i] - y[:, j]))
                d2 = cost_matrix[i - 1][j] + _cost_dependent(
                    x[:, i], x[:, i - 1], y[:, j], c
                )
                d3 = cost_matrix[i][j - 1] + _cost_dependent(
                    y[:, j], x[:, i], y[:, j - 1], c
                )

                cost_matrix[i, j] = _softmin3(d1, d2, d3, gamma)
    return cost_matrix


def soft_msm_pairwise_distance(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    window: Optional[float] = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    multivariate_conversion = _is_multivariate(X, y)
    _X, unequal_length = _convert_to_list(X, "X", multivariate_conversion)

    if y is None:
        return _soft_msm_pairwise_distance(
            _X, window, independent, c, itakura_max_slope, gamma, unequal_length
        )

    _y, unequal_length = _convert_to_list(y, "y", multivariate_conversion)
    return _soft_msm_from_multiple_to_multiple_distance(
        _X, _y, window, independent, c, itakura_max_slope, gamma, unequal_length
    )


@njit(cache=True, fastmath=True)
def _soft_msm_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    independent: bool,
    c: float,
    itakura_max_slope: Optional[float],
    gamma: float,
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )
    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_msm_distance(
                x1, x2, bounding_matrix, independent, c, gamma
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _soft_msm_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    independent: bool,
    c: float,
    itakura_max_slope: Optional[float],
    gamma: float,
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )
    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_msm_distance(
                x1, y1, bounding_matrix, independent, c, gamma
            )
    return distances


@njit(cache=True, fastmath=True)
def soft_msm_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    gamma: float = 1.0,
) -> Tuple[List[Tuple[int, int]], float]:
    cost_matrix = soft_msm_cost_matrix(
        x, y, window, independent, c, itakura_max_slope, gamma
    )
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
