import numpy as np
from sklearn.utils import check_random_state

from aeon.clustering._kesba_funcs import _first, _kesba_kmeans_plus_plus, kesba
from aeon.testing.data_generation import make_example_3d_numpy

if __name__ == "__main__":
    X_train = make_example_3d_numpy(100, 1, 100, random_state=1, return_y=False)
    n_clusters = 6
    window = 1.0

    # init_temp = _first(X_train, n_clusters, 0.2)
    init_temp = "ours"

    labels, cluster_centers, inertia, _ = kesba(
        X=X_train.copy(),
        n_clusters=n_clusters,
        random_state=1,
        algorithm="ours",
        init=init_temp,
        window=window,
        verbose=True,
    )
    print("++++++++++++++++++++++++++++++++++++++")
    labels_lloyds, lloyds_cluster_centers, inertia_lloyds, _ = kesba(
        X=X_train.copy(),
        n_clusters=n_clusters,
        random_state=1,
        algorithm="lloyds",
        init=init_temp,
        window=window,
        verbose=True,
    )
    print("++++++++++++++++++++++++++++++++++++++")
    print("Are labels the same? ", np.array_equal(labels, labels_lloyds))
    print("Inertia: ", inertia)
    print("Inertia lloyds: ", inertia_lloyds)
    print("Inertia difference: ", inertia - inertia_lloyds)
    print(
        "Are cluster centers the same? ",
        np.array_equal(cluster_centers, lloyds_cluster_centers),
    )
