"""Temp kmeans script."""

import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

from aeon.clustering import KESBA
from aeon.datasets import load_acsf1, load_gunpoint
from aeon.testing.data_generation import make_example_3d_numpy

if __name__ == "__main__":

    # X_train, y_train = load_acsf1(split="train")
    X_train, y_train = load_gunpoint(split="train")
    n_clusters = len(set(list(y_train)))
    # X_train = make_example_3d_numpy(n_cases=100, n_channels=1, n_timepoints=100, return_y=False)
    # n_clusters = 5
    max_iters = 20
    window = 0.2
    ba_subset_size = 0.2

    # distance = "twe"
    # averaging_method = "random_subset_ssg"
    distance = "msm"
    averaging_method = "random_subset_ssg"
    verbose = True

    # Tony implementation
    print("++++++++++++++++++Tony Elkan++++++++++++++++++++")
    clst = KESBA(
        n_clusters=n_clusters,
        distance=distance,
        window=window,
        ba_subset_size=ba_subset_size,
        max_iter=max_iters,
        random_state=1,
        averaging_method=averaging_method,
        algorithm="tony-elkan",
        verbose=verbose,
    )

    start = time.time()
    clst.fit(X_train)
    end = time.time()
    print("Time to fit with Tony Elkan: ", end - start)
    print("Tony Elkan Number of distance calls: ", clst.num_distance_calls)
    tony_elkan_labels = clst.labels_
    print("++++++++++++++++++Tony Elkan++++++++++++++++++++")

    # print("++++++++++++++++++Elkan++++++++++++++++++++")
    #
    # clst = KESBA(
    #     n_clusters=n_clusters,
    #     distance=distance,
    #     window=window,
    #     ba_subset_size=ba_subset_size,
    #     max_iter=max_iters,
    #     random_state=1,
    #     averaging_method=averaging_method,
    #     algorithm="elkan",
    #     verbose=verbose,
    # )
    #
    # start = time.time()
    # clst.fit(X_train)
    # end = time.time()
    # print("Time to fit with Elkan: ", end - start)
    # print("Elkan Number of distance calls: ", clst.num_distance_calls)
    # print(
    #     "Skip 1 ",
    #     clst.skip1,
    #     " skip 2 =",
    #     clst.skip2,
    #     " skip 3 = ",
    #     clst.skip3,
    #     " skip 4 = ",
    #     clst.skip4,
    # )
    #
    # elkan_labels = clst.labels_
    # print("Num clusters", len(set(elkan_labels)))

    print("++++++++++++++++++Lloyds++++++++++++++++++++")

    clst = KESBA(
        n_clusters=n_clusters,
        distance=distance,
        window=window,
        ba_subset_size=ba_subset_size,
        max_iter=max_iters,
        random_state=1,
        averaging_method=averaging_method,
        algorithm="lloyds",
        verbose=verbose,
    )

    start = time.time()
    clst.fit(X_train)
    end = time.time()
    print("Time to fit with Lloyds: ", end - start)
    print("Lloyds Number of distance calls: ", clst.num_distance_calls)

    print("++++++++++++++++++Lloyds++++++++++++++++++++")

    lloyds_labels = clst.labels_

    # print("Eklan ARI: ", adjusted_rand_score(y_train, elkan_labels))
    # print("Lloyds ARI: ", adjusted_rand_score(y_train, lloyds_labels))

    # print("Elkan labels: ", elkan_labels)
    # print("Lloyds labels: ", lloyds_labels)
    # print("Are Elkan -> lloyds the labels the same? ", np.array_equal(elkan_labels, lloyds_labels))
    print("Are Tony Elkan -> lloyds the labels the same? ", np.array_equal(tony_elkan_labels, lloyds_labels))
