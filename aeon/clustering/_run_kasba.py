from sklearn.metrics import adjusted_rand_score
from statsmodels.tsa.statespace.simulation_smoother import check_random_state

from aeon.clustering import KASBA, KESBA
from aeon.datasets import load_gunpoint
from aeon.distances import pairwise_distance


def _random_init(X, n_clusters, random_state):
    _random_state = check_random_state(random_state)
    cluster_centres = X[_random_state.choice(X.shape[0], n_clusters, replace=False)]
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
    X_train, y_train = load_gunpoint(split="train")
    n_clusters = len(set(list(y_train)))

    init = _random_init(X_train, n_clusters, 1)

    kasba_clust = KASBA(n_clusters=n_clusters, init=init)

    kesba_clust = KESBA(n_clusters=n_clusters, init=init)

    kesba_labels = kesba_clust.fit_predict(X_train)
    kasba_labels = kasba_clust.fit_predict(X_train)

    print("KESBA ARI: ", adjusted_rand_score(y_train, kesba_labels))
    print("KASBA ARI: ", adjusted_rand_score(y_train, kasba_labels))
