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
    soft_dtw_alignment_path,
    soft_dtw_cost_matrix,
    soft_dtw_distance,
    soft_dtw_pairwise_distance,
)
from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)
from aeon.transformations.tests.test_pipeline import X_train

if __name__ == "__main__":
    X = make_example_3d_numpy(
        n_timepoints=10, n_channels=1, n_cases=20, random_state=1, return_y=False
    )
    tslearn_X = X.swapaxes(1, 2)

    distance_params = {"gamma": 0.5, "window": 0.2}

    average_params = {
        **distance_params,
        "method": "soft_dba",
    }

    km = TimeSeriesKMeans(
        n_clusters=2,
        averaging_method="ba",
        distance="soft_dtw",
        distance_params=distance_params,
        average_params=average_params,
    )
    km.fit(X)

    #
    # aeon_dist = soft_dtw_distance(x, y, gamma=1.0)
    # aeon_cm = soft_dtw_cost_matrix(x, y, gamma=1.0)
    # aeon_ap = soft_dtw_alignment_path(x, y, gamma=1.0)
    #
    # aeon_cm_window = soft_dtw_cost_matrix(x, y, gamma=1.0, window=0.2)
    # stop = ""
    #
    # general_dist = distance(x, y, "soft_dtw", gamma=1.0)
    # general_cm = cost_matrix(x, y, "soft_dtw", gamma=1.0)
    # general_ap = alignment_path(x, y, "soft_dtw", gamma=1.0)
    #
    # assert np.allclose(aeon_dist, general_dist)
    # assert np.allclose(aeon_cm, general_cm)
    # assert np.allclose(aeon_ap[1], general_ap[1])
    #
    # for general, specific in zip(general_ap[0], aeon_ap[0]):
    #     assert general == specific
    #
    # pw_X = make_example_3d_numpy(
    #     n_timepoints=10, n_channels=1, n_cases=20, random_state=1, return_y=False
    # )
    # tslearn_X = pw_X.swapaxes(1, 2)
    # pw_other = make_example_3d_numpy(
    #     n_timepoints=10, n_channels=1, n_cases=20, random_state=2, return_y=False
    # )
    #
    # aeon_pw_self = soft_dtw_pairwise_distance(pw_X, gamma=1.0)
    # aeon_pw = soft_dtw_pairwise_distance(pw_X, pw_other, gamma=1.0)
    #
    # general_pw_self = pairwise_distance(pw_X, metric="soft_dtw", gamma=1.0)
    # general_pw = pairwise_distance(pw_X, pw_other, metric="soft_dtw", gamma=1.0)
    #
    # assert np.allclose(aeon_pw_self, general_pw_self)
    # assert np.allclose(aeon_pw, general_pw)
    #
    # from aeon.clustering.averaging._ba_soft_dtw import soft_dtw_barycenter_average
    #
    # # average_soft_dba = elastic_barycenter_average(pw_X, distance="soft_dtw", gamma=1.0)
    # average_soft_dba = soft_dtw_barycenter_average(pw_X, gamma=1.0)
    #
    # from tslearn.barycenters import softdtw_barycenter
    #
    # average_tslearn = softdtw_barycenter(tslearn_X, gamma=1.0)
    #
    # # swap back
    # average_tslearn = average_tslearn.swapaxes(0, 1)
    #
    # close = np.allclose(average_soft_dba, average_tslearn)
    #
    # stop = ""

    # tslearn_x = x.swapaxes(0, 1)
    # tslearn_y = y.swapaxes(0, 1)
    #
    # tslearn_dist = soft_dtw(tslearn_x, tslearn_y, gamma=0.1)
    #
    # X_train, y_train = load_gunpoint(return_X_y=True)
    #
    # tslearn_X_train = X_train.swapaxes(1, 2)
    # import time
    #
    # aeon_start = time.time()
    # aeon_pw = soft_dtw_pairwise_distance(X_train)
    # aeon_end = time.time()
    # aeon_time = aeon_end - aeon_start
    #
    # tslearn_start = time.time()
    # tslearn_pw = pairwise_distance(tslearn_X_train, metric=soft_dtw)
    # tslearn_end = time.time()
    # tslearn_time = tslearn_end - tslearn_start
    # import numpy as np
    #
    # # Check every result is equal
    # for i in range(len(aeon_pw)):
    #     for j in range(len(aeon_pw)):
    #         assert np.isclose(aeon_pw[i, j], tslearn_pw[i, j])
    #
    # print(f"aeon time: {aeon_time}")
    # print(f"tslearn time: {tslearn_time}")
    #
    # aeon_start_window = time.time()
    # aeon_pw = soft_dtw_pairwise_distance(X_train, window=0.2)
    # aeon_end_window = time.time()
    # aeon_time = aeon_end_window - aeon_start_window
    #
    # print(f"aeon window time: {aeon_time}")
    # stop = ""
