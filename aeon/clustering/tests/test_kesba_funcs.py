import pytest
from sklearn.utils import check_random_state

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.clustering._kesba import _elastic_kmeans_plus_plus
from aeon.distances import create_bounding_matrix

# n_cases = [10, 50, 100, 200]
# n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_cases = [20]
n_clusters = [3]
n_timepoints = 10
window = 0.1
random_state = check_random_state(1)

@pytest.mark.parametrize("n_cluster", n_clusters)
@pytest.mark.parametrize("n_case", n_cases)
def test_kmeans_plus_plus_init_method(n_cluster, n_case):
    boudning_matrix = create_bounding_matrix(n_timepoints, n_timepoints, window=window)
    X_train = make_example_3d_numpy(n_case, 1, n_timepoints, random_state=1, return_y=False)
    cluster_centers, distances_to_centres, labels = _elastic_kmeans_plus_plus(
        X=X_train,
        random_state=random_state,
        n_clusters=n_cluster,
        bounding_matrix=boudning_matrix,
        c=1.0,
    )
    assert len(labels) == n_case
    assert len(cluster_centers) == n_cluster
    assert cluster_centers.shape == (n_cluster, X_train.shape[1], X_train.shape[2])
    assert distances_to_centres.shape == (n_case,)
