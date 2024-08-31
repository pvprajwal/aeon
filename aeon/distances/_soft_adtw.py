r"""Soft Amercing dynamic time warping (ADTW) between two time series."""

__maintainer__ = []

from typing import List, Optional, Tuple, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import _univariate_squared_distance
from aeon.distances._adtw import _adtw_cost_matrix
from aeon.distances._utils import _convert_to_list, _is_multivariate, _softmin3


@njit(cache=True, fastmath=True)
def soft_adtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
    gamma: float = 1.0,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_adtw_distance(_x, _y, bounding_matrix, warp_penalty, gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_adtw_distance(x, y, bounding_matrix, warp_penalty, gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_adtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_adtw_cost_matrix(_x, _y, bounding_matrix, warp_penalty, gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_adtw_cost_matrix(x, y, bounding_matrix, warp_penalty, gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_adtw_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, warp_penalty: float, gamma: float
) -> float:
    return _soft_adtw_cost_matrix(x, y, bounding_matrix, warp_penalty, gamma)[
        x.shape[1] - 1, y.shape[1] - 1
    ]


@njit(cache=True, fastmath=True)
def _soft_adtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, warp_penalty: float, gamma: float
) -> np.ndarray:
    if gamma == 0.0:
        return _adtw_cost_matrix(x, y, bounding_matrix, warp_penalty)
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_squared_distance(
                    x[:, i], y[:, j]
                ) + _softmin3(
                    cost_matrix[i, j + 1] + warp_penalty,
                    cost_matrix[i + 1, j] + warp_penalty,
                    cost_matrix[i, j],
                    gamma
                )

    return cost_matrix[1:, 1:]


def soft_adtw_pairwise_distance(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    multivariate_conversion = _is_multivariate(X, y)
    _X, unequal_length = _convert_to_list(X, "X", multivariate_conversion)
    if y is None:
        # To self
        return _soft_adtw_pairwise_distance(
            _X, window, itakura_max_slope, warp_penalty, gamma, unequal_length
        )

    _y, unequal_length = _convert_to_list(y, "y", multivariate_conversion)
    return _soft_adtw_from_multiple_to_multiple_distance(
        _X, _y, window, itakura_max_slope, warp_penalty, gamma, unequal_length
    )


@njit(cache=True, fastmath=True)
def _soft_adtw_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    itakura_max_slope: Optional[float],
    warp_penalty: float,
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
            distances[i, j] = _soft_adtw_distance(x1, x2, bounding_matrix, warp_penalty, gamma)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _soft_adtw_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    itakura_max_slope: Optional[float],
    warp_penalty: float,
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
            distances[i, j] = _soft_adtw_distance(x1, y1, bounding_matrix, warp_penalty, gamma)
    return distances


@njit(cache=True, fastmath=True)
def soft_adtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
    gamma: float = 1.0,
) -> Tuple[List[Tuple[int, int]], float]:
    cost_matrix = soft_adtw_cost_matrix(x, y, window, itakura_max_slope, warp_penalty, gamma)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
