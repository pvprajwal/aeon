import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.clustering import KESBA

use_lloyds = [True, False]

@pytest.mark.parametrize("use_lloyds", use_lloyds)
def test_kesba(use_lloyds):
    X_train = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)
    X_test = make_example_3d_numpy(10, 1, 10, random_state=2, return_y=False)
    clst = KESBA(n_clusters=4, random_state=1, use_lloyds=use_lloyds)
    clst.fit(X_train)

    assert clst.labels_.shape == (10,)
    assert clst.cluster_centers_.shape == (4, X_train.shape[1], X_train.shape[2])
    assert isinstance(clst.inertia_, float)
    assert clst.n_iter_ > 0

    preds = clst.predict(X_test)
    assert preds.shape == (10,)


def test_kesba_equivalence():
    X_train = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)
    X_test = make_example_3d_numpy(10, 1, 10, random_state=2, return_y=False)
    clst = KESBA(n_clusters=4, random_state=1, use_lloyds=False)
    clst_ll = KESBA(n_clusters=4, random_state=1, use_lloyds=True)
    clst.fit(X_train)
    clst_ll.fit(X_train)

    assert np.array_equal(clst.labels_, clst_ll.labels_)
    assert np.array_equal(clst.cluster_centers_, clst_ll.cluster_centers_)
    assert clst.inertia_ == clst_ll.inertia_

    preds = clst.predict(X_test)
    preds_lloyds = clst_ll.predict(X_test)
    assert np.array_equal(preds, preds_lloyds)

def test_distance_params():
    X_train = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)
    X_test = make_example_3d_numpy(10, 1, 10, random_state=2, return_y=False)
    clst = KESBA(n_clusters=4, random_state=1, use_lloyds=False, distance_params={"c": 2})
    clst.fit(X_train)
    preds = clst.predict(X_test)

    clst = KESBA(n_clusters=4, random_state=1, use_lloyds=False, distance_params={"c": 10})
    clst.fit(X_train)
    pred_other = clst.predict(X_test)

    assert not np.array_equal(preds, pred_other)

def test_different_distances():
    X_train = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)
    X_test = make_example_3d_numpy(10, 1, 10, random_state=2, return_y=False)
    clst = KESBA(n_clusters=4, random_state=1, use_lloyds=False, distance="msm")
    clst.fit(X_train)
    preds = clst.predict(X_test)
    cluster_centers = clst.cluster_centers_

    clst = KESBA(n_clusters=4, random_state=1, use_lloyds=False, distance="twe")
    clst.fit(X_train)
    pred_other = clst.predict(X_test)
    cluster_centers_other = clst.cluster_centers_

    assert not np.array_equal(preds, pred_other)
    assert not np.array_equal(cluster_centers, cluster_centers_other)