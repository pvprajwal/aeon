import numpy as np
from tslearn.clustering import TimeSeriesKMeans as tslearn_KMeans
from tslearn.metrics import soft_dtw

from aeon.clustering import TimeSeriesKMeans
from aeon.clustering.averaging import elastic_barycenter_average
from aeon.datasets import load_gunpoint
from aeon.distances import (
    alignment_path,
    cost_matrix,
    distance,
    pairwise_distance,
    soft_adtw_alignment_path,
    soft_adtw_cost_matrix,
    soft_adtw_distance,
    soft_adtw_pairwise_distance,
    soft_msm_alignment_path,
    soft_msm_cost_matrix,
    soft_msm_distance,
    soft_msm_pairwise_distance,
    soft_twe_alignment_path,
    soft_twe_cost_matrix,
    soft_twe_distance,
    soft_twe_pairwise_distance,
)
from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)
from aeon.transformations.tests.test_pipeline import X_train

def check_distance_works(metric: str, dist_func, cost_matrix_func, align_path_func, pw_func):
    aeon_dist = dist_func(x, y, gamma=1.0)
    aeon_cm = cost_matrix_func(x, y, gamma=1.0)
    aeon_ap = align_path_func(x, y, gamma=1.0)

    general_dist = distance(x, y, metric, gamma=1.0)
    general_cm = cost_matrix(x, y, metric, gamma=1.0)
    general_ap = alignment_path(x, y, metric, gamma=1.0)

    assert np.allclose(aeon_dist, general_dist)
    assert np.allclose(aeon_cm, general_cm)
    assert np.allclose(aeon_ap[1], general_ap[1])

    for general, specific in zip(general_ap[0], aeon_ap[0]):
        assert general == specific

    pw_X = make_example_3d_numpy(
        n_timepoints=10, n_channels=1, n_cases=20, random_state=1, return_y=False
    )
    pw_other = make_example_3d_numpy(
        n_timepoints=10, n_channels=1, n_cases=20, random_state=2, return_y=False
    )

    aeon_pw_self = pw_func(pw_X, gamma=1.0)
    aeon_pw = pw_func(pw_X, pw_other, gamma=1.0)

    general_pw_self = pairwise_distance(pw_X, metric=metric, gamma=1.0)
    general_pw = pairwise_distance(pw_X, pw_other, metric=metric, gamma=1.0)

    assert np.allclose(aeon_pw_self, general_pw_self)
    assert np.allclose(aeon_pw, general_pw)

    gamma_zero = dist_func(x, y, gamma=0.0)
    original_dist = distance(x, y, metric=metric.replace("soft_", ""))

    assert np.allclose(gamma_zero, original_dist)

if __name__ == "__main__":
    X = make_example_3d_numpy(
        n_timepoints=10, n_channels=1, n_cases=20, random_state=1, return_y=False
    )
    x = make_example_2d_numpy_series(n_timepoints=10, n_channels=1, random_state=1)
    y = make_example_2d_numpy_series(n_timepoints=10, n_channels=1, random_state=2)

    check_distance_works("soft_adtw", soft_adtw_distance, soft_adtw_cost_matrix, soft_adtw_alignment_path, soft_adtw_pairwise_distance)
    check_distance_works("soft_msm", soft_msm_distance, soft_msm_cost_matrix, soft_msm_alignment_path, soft_msm_pairwise_distance)
    check_distance_works("soft_twe", soft_twe_distance, soft_twe_cost_matrix, soft_twe_alignment_path, soft_twe_pairwise_distance)



