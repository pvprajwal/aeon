"""Temp kmeans script."""
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

from aeon.clustering import TimeSeriesKMeans
from aeon.datasets import load_acsf1, load_gunpoint
from aeon.testing.data_generation import make_example_3d_numpy

if __name__ == "__main__":

    X_train, y_train = load_acsf1(split="train")
    # X_train = make_example_3d_numpy(n_cases=1000, n_channels=1, n_timepoints=1000, return_y=False)

    n_clusters = len(set(list(y_train)))
    distance = "euclidean"
    verbose = True

    clst = TimeSeriesKMeans(
        n_clusters=n_clusters,
        distance=distance,
        n_init=1,
        max_iter=300,
        random_state=0,
        averaging_method="mean",
        algorithm="elkan",
        verbose=verbose,
    )

    start = time.time()
    clst.fit(X_train)
    end = time.time()
    print("Time to fit with Elkan: ", end - start)
    print("Elkan Number of distance calls: ", clst.num_distance_calls)

    elkan_labels = clst.labels_
    print("Num clusters", len(set(elkan_labels)))

    clst = TimeSeriesKMeans(
        n_clusters=n_clusters,
        distance=distance,
        n_init=1,
        max_iter=300,
        random_state=0,
        averaging_method="mean",
        algorithm="lloyds",
        verbose=verbose,
    )

    start = time.time()
    clst.fit(X_train)
    end = time.time()
    print("Time to fit with Lloyds: ", end - start)
    print("Lloyds Number of distance calls: ", clst.num_distance_calls)

    lloyds_labels = clst.labels_

    print("Eklan ARI: ", adjusted_rand_score(y_train, elkan_labels))
    print("Lloyds ARI: ", adjusted_rand_score(y_train, lloyds_labels))

    # print("Elkan labels: ", elkan_labels)
    # print("Lloyds labels: ", lloyds_labels)
    print("Are the labels the same? ", np.array_equal(elkan_labels, lloyds_labels))
