import numpy as np
from numba import njit
from sklearn.metrics import adjusted_rand_score

from aeon.clustering import KASBA, KESBA
from aeon.clustering._kasba_numba import KASBA_NUMBA
from aeon.datasets import load_acsf1, load_gunpoint
from aeon.distances import pairwise_distance


@njit(nogil=True, cache=True)
def _first_init(X, n_clusters):
    cluster_centres = X[0:n_clusters]
    pw_dists = pairwise_distance(
        X,
        cluster_centres,
        metric="msm",
        **{"c": 1.0},
    )
    min_dists = pw_dists.min(axis=1)
    labels = pw_dists.argmin(axis=1)
    return cluster_centres, min_dists, labels


if __name__ == "__main__":
    # X_train, y_train = load_gunpoint(split="train")
    X_train, y_train = load_acsf1(split="train")
    n_clusters = len(set(list(y_train)))
    verbose = False
    debug_deterministic = True

    kasba_clust = KASBA(
        n_clusters=n_clusters,
        random_state=1,
    )

    kesba_clust = KESBA(
        n_clusters=n_clusters,
        random_state=1,
        verbose=verbose,
    )

    kasba_numba_clust = KASBA_NUMBA(
        n_clusters=n_clusters,
        random_state=1,
        verbose=verbose,
    )
    print("Starting")
    kasba_numba_clust._create_numba_caches(X_train)

    print("\n=========== KASBA =========")
    kasba_labels = kasba_clust.fit_predict(X_train)
    print("\n=========== KESBA =========")
    kesba_labels = kesba_clust.fit_predict(X_train)
    print("\n=========== KASBA NUMBA =========")
    kasba_numba_labels = kasba_numba_clust.fit_predict(X_train)

    print("KESBA ARI: ", adjusted_rand_score(y_train, kesba_labels))
    print("KASBA ARI: ", adjusted_rand_score(y_train, kasba_labels))
    print("KASBA NUMBA ARI: ", adjusted_rand_score(y_train, kasba_numba_labels))

    print(f"Labels match {np.array_equal(kesba_labels, kasba_labels)}")
    print(f"Labels match {np.array_equal(kesba_labels, kasba_numba_labels)}")
