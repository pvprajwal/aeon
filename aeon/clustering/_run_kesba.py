"""Temp kmeans script."""

import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

from aeon.clustering import KESBA
from aeon.datasets import load_acsf1, load_gunpoint
from aeon.testing.data_generation import make_example_3d_numpy

max_iters = 100
window = 0.2
ba_subset_size = 0.2
distance = "msm"
# averaging_method = "petitjean"
averaging_method = "random_subset_ssg"
# averaging_method = "subgradient"
verbose = True
init = "first"
random_state = 1


# def run_eklan_kesba(X_train, n_clusters):
#     print("++++++++++++++++++Elkan+++++++++++++++++++++")
#
#     clst = KESBA_old(
#         n_clusters=n_clusters,
#         init=init,
#         distance=distance,
#         window=window,
#         ba_subset_size=ba_subset_size,
#         max_iter=max_iters,
#         random_state=1,
#         averaging_method=averaging_method,
#         algorithm="elkan",
#         verbose=verbose,
#     )
#
#     start = time.time()
#     clst.fit(X_train)
#     end = time.time()
#     print("Converged at iteration: ", clst.n_iter_)
#     print("Time to fit with Elkan: ", end - start)
#     print("Elkan Number of distance calls: ", clst.num_distance_calls)
#     print("++++++++++++++++++Elkan+++++++++++++++++++++")
#     return clst.labels_


def run_lloyds_kesba(X_train, n_clusters):
    print("++++++++++++++++++Lloyds++++++++++++++++++++")

    clst = KESBA(
        n_clusters=n_clusters,
        distance=distance,
        ba_subset_size=ba_subset_size,
        window=window,
        max_iter=max_iters,
        random_state=random_state,
        average_method=averaging_method,
        verbose=verbose,
        use_lloyds=True,
    )

    start = time.time()
    clst.fit(X_train)
    end = time.time()
    print("Converged at iteration: ", clst.n_iter_)
    print("Time to fit with Lloyds: ", end - start)

    print("++++++++++++++++++Lloyds++++++++++++++++++++")
    return clst.labels_, clst.inertia_


def run_tony_kesba(X_train, n_clusters):
    print("++++++++++++++++Tony Elkan++++++++++++++++++")
    clst = KESBA(
        n_clusters=n_clusters,
        distance=distance,
        ba_subset_size=ba_subset_size,
        window=window,
        max_iter=max_iters,
        random_state=random_state,
        average_method=averaging_method,
        verbose=verbose,
        use_lloyds=False,
    )

    start = time.time()
    clst.fit(X_train)
    end = time.time()
    print("Converged at iteration: ", clst.n_iter_)
    print("Time to fit with Tony Elkan: ", end - start)
    print("++++++++++++++++Tony Elkan++++++++++++++++++")
    return clst.labels_, clst.inertia_


if __name__ == "__main__":
    run_lloyds = True
    run_elkan = False
    run_tony = True

    X_train, y_train = load_gunpoint(split="train")
    # n_clusters = 5
    # X_train, y_train = load_acsf1(split="train")
    # X_train = make_example_3d_numpy(n_cases=100, n_channels=1, n_timepoints=100, return_y=False, random_state=1)
    n_clusters = len(set(list(y_train)))
    # n_clusters = 5

    if run_tony:
        tony_elkan_labels, tony_elkan_inertia = run_tony_kesba(X_train, n_clusters)
    if run_lloyds:
        lloyds_labels, lloyds_inertia = run_lloyds_kesba(X_train, n_clusters)

    # if run_elkan:
    #     elkan_labels = run_eklan_kesba(X_train, n_clusters)

    # if run_lloyds and run_elkan:
    #     print(
    #         "Are Elkan -> lloyds the labels the same? ",
    #         np.array_equal(elkan_labels, lloyds_labels),
    #     )

    if run_tony and run_lloyds:
        print(
            "Are Tony Elkan -> lloyds the labels the same? ",
            np.array_equal(tony_elkan_labels, lloyds_labels),
        )
        print(
            "Are Tony Elkan -> lloyds the inertia the same? ",
            tony_elkan_inertia == lloyds_inertia,
        )
        if not tony_elkan_inertia == lloyds_inertia:
            print("Tony Elkan inertia: ", tony_elkan_inertia)
            print("Lloyds inertia: ", lloyds_inertia)
