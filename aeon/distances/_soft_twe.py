from typing import List, Optional, Tuple, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._euclidean import _univariate_euclidean_distance
from aeon.distances._utils import _convert_to_list, _is_multivariate, _softmin3
from aeon.distances._twe import _twe_cost_matrix, _pad_arrs


@njit(cache=True, fastmath=True)
def soft_twe_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    gamma: float = 1.0,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_distance(_pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda, gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_distance(_pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda, gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_twe_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_cost_matrix(_pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda, gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_cost_matrix(_pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda, gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_twe_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, nu: float, lmbda: float, gamma: float
) -> float:
    return _soft_twe_cost_matrix(x, y, bounding_matrix, nu, lmbda, gamma)[
        x.shape[1] - 2, y.shape[1] - 2
    ]


@njit(cache=True, fastmath=True)
def _soft_twe_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, nu: float, lmbda: float, gamma: float
) -> np.ndarray:
    if gamma == 0.0:
        return _twe_cost_matrix(x, y, bounding_matrix, nu, lmbda)
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = 0.0

    del_add = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i - 1, j - 1]:
                del_x_squared_dist = _univariate_euclidean_distance(
                    x[:, i - 1], x[:, i]
                )
                del_x = cost_matrix[i - 1, j] + del_x_squared_dist + del_add

                del_y_squared_dist = _univariate_euclidean_distance(
                    y[:, j - 1], y[:, j]
                )
                del_y = cost_matrix[i, j - 1] + del_y_squared_dist + del_add

                match_same_squared_d = _univariate_euclidean_distance(x[:, i], y[:, j])
                match_prev_squared_d = _univariate_euclidean_distance(
                    x[:, i - 1], y[:, j - 1]
                )
                match = (
                    cost_matrix[i - 1, j - 1]
                    + match_same_squared_d
                    + match_prev_squared_d
                    + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                )

                cost_matrix[i, j] = _softmin3(del_x, del_y, match, gamma)

    return cost_matrix[1:, 1:]


def soft_twe_pairwise_distance(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    multivariate_conversion = _is_multivariate(X, y)
    _X, unequal_length = _convert_to_list(X, "X", multivariate_conversion)
    if y is None:
        return _soft_twe_pairwise_distance(
            _X, window, nu, lmbda, itakura_max_slope, gamma, unequal_length
        )
    _y, unequal_length = _convert_to_list(y, "y", multivariate_conversion)
    return _soft_twe_from_multiple_to_multiple_distance(
        _X, _y, window, nu, lmbda, itakura_max_slope, gamma, unequal_length
    )


@njit(cache=True, fastmath=True)
def _soft_twe_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    nu: float,
    lmbda: float,
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

    padded_X = NumbaList()
    for i in range(n_cases):
        padded_X.append(_pad_arrs(X[i]))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = padded_X[i], padded_X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_twe_distance(x1, x2, bounding_matrix, nu, lmbda, gamma)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _soft_twe_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    nu: float,
    lmbda: float,
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

    padded_x = NumbaList()
    for i in range(n_cases):
        padded_x.append(_pad_arrs(x[i]))

    padded_y = NumbaList()
    for i in range(m_cases):
        padded_y.append(_pad_arrs(y[i]))

    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = padded_x[i], padded_y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_twe_distance(x1, y1, bounding_matrix, nu, lmbda, gamma)
    return distances


@njit(cache=True, fastmath=True)
def soft_twe_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    gamma: float = 1.0,
) -> Tuple[List[Tuple[int, int]], float]:
    cost_matrix = soft_twe_cost_matrix(x, y, window, nu, lmbda, itakura_max_slope, gamma)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
