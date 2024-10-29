import numpy as np
from sklearn.utils import check_random_state

from aeon.clustering._k_means import EmptyClusterError
from aeon.clustering.averaging import elastic_barycenter_average
from aeon.distances import msm_distance, msm_pairwise_distance


# ==================== KESBA ====================
def kesba(
    X,
    n_clusters,
    random_state,
    algorithm,
    init,
    window,
    verbose,
):
    random_state = check_random_state(random_state)
    if isinstance(init, tuple):
        temp_cluster_centres, temp_distances_to_centres, temp_labels = init
        cluster_centres = temp_cluster_centres.copy()
        distances_to_centres = temp_distances_to_centres.copy()
        labels = temp_labels.copy()
    elif init == "first":
        cluster_centres, distances_to_centres, labels = _first(
            X,
            n_clusters,
            window,
        )
    else:
        cluster_centres, distances_to_centres, labels = _kesba_kmeans_plus_plus(
            X,
            random_state,
            n_clusters,
            window,
        )

    print("Starting inertia: ", np.sum(distances_to_centres**2))

    if algorithm == "lloyds":
        return _kesba_lloyds(
            X,
            cluster_centres,
            distances_to_centres,
            labels,
            n_clusters,
            random_state,
            window,
            verbose,
        )
    else:
        return _kesba(
            X,
            cluster_centres,
            distances_to_centres,
            labels,
            n_clusters,
            random_state,
            window,
            verbose,
        )


# ==================== KESBA ====================

# curr_pw_1 = msm_pairwise_distance(X, cluster_centres, window=window)
# labels_1 = curr_pw_1.argmin(axis=1)
# distances_to_centres_1 = curr_pw_1.min(axis=1)
# inertia_1 = np.sum(distances_to_centres_1**2)
#
# labels_equal = np.array_equal(labels, labels_1)
# inertia_equal = inertia == inertia_1
# index_diff = []
#
# for index_i in range(X.shape[0]):
#     if labels[index_i] != labels_1[index_i]:
#         index_diff.append(index_i)
#
# num_diff = len(index_diff)


def _kesba(
    X,
    cluster_centres,
    distances_to_centres,
    labels,
    n_clusters,
    random_state,
    window,
    verbose,
):
    prev_labels = None
    prev_cluster_centres = None
    for i in range(100):
        distance_between_centres = msm_pairwise_distance(
            cluster_centres,
            cluster_centres,
            window=window,
        )

        labels, distances_to_centres, inertia = _kesba_assignment(
            X,
            cluster_centres,
            distances_to_centres,
            labels,
            i == 0,
            distance_between_centres,
            n_clusters,
            window,
            verbose,
        )

        labels, cluster_centres, distances_to_centres = _handle_empty_cluster(
            X,
            cluster_centres,
            distances_to_centres,
            labels,
            n_clusters,
            window,
        )

        if np.array_equal(prev_labels, labels):
            # if change_in_centres < self.tol:
            if verbose:
                print(  # noqa: T001
                    f"Converged at iteration {i}, inertia {inertia:.5f}."
                )
            break

        prev_inertia = inertia
        prev_labels = labels.copy()
        prev_cluster_centres = cluster_centres.copy()

        cluster_centres, distances_to_centres = _kesba_update(
            X,
            cluster_centres,
            labels,
            n_clusters,
            random_state,
            window,
            i,
            distances_to_centres,
        )

        if verbose is True:
            print(f"Iteration {i}, inertia {prev_inertia}.")  # noqa: T001, T201

    final_inertia = prev_inertia
    final_labels = prev_labels
    final_cluster_centres = prev_cluster_centres

    if inertia < prev_inertia:
        final_inertia = inertia
        final_labels = labels
        final_cluster_centres = cluster_centres

    return final_labels, final_cluster_centres, final_inertia, i + 1


def _kesba_lloyds(
    X,
    cluster_centres,
    distances_to_centres,
    labels,
    n_clusters,
    random_state,
    window,
    verbose,
):
    prev_inertia = np.inf
    prev_labels = None
    prev_cluster_centres = None
    for i in range(100):

        curr_pw = msm_pairwise_distance(X, cluster_centres, window=window)
        labels = curr_pw.argmin(axis=1)
        distances_to_centres = curr_pw.min(axis=1)
        inertia = np.sum(distances_to_centres**2)

        labels, cluster_centres, distances_to_centres = _handle_empty_cluster(
            X,
            cluster_centres,
            distances_to_centres,
            labels,
            n_clusters,
            window,
        )

        if verbose:
            print(f"{inertia:.5f}", end=" --> ")

        if np.array_equal(prev_labels, labels):
            # if change_in_centres < self.tol:
            if verbose:
                print(  # noqa: T001
                    f"Converged at iteration {i}, inertia {inertia:.5f}."
                )

            break

        prev_inertia = inertia
        prev_labels = labels.copy()
        prev_cluster_centres = cluster_centres.copy()

        # Compute new cluster centres
        cluster_centres, distances_to_centres = _kesba_update(
            X,
            cluster_centres,
            labels,
            n_clusters,
            random_state,
            window,
            i,
            distances_to_centres,
        )

        if verbose is True:
            print(f"Iteration {i}, inertia {prev_inertia}.")  # noqa: T001, T201

    final_inertia = prev_inertia
    final_labels = prev_labels
    final_cluster_centres = prev_cluster_centres

    if inertia < prev_inertia:
        final_inertia = inertia
        final_labels = labels
        final_cluster_centres = cluster_centres

    return final_labels, final_cluster_centres, final_inertia, i + 1


# ==================== Assignment ====================
def _kesba_assignment(
    X,
    cluster_centres,
    distances_to_centres,
    labels,
    is_first_iteration,
    distances_between_centres,
    n_clusters,
    window,
    verbose,
):
    for i in range(X.shape[0]):
        min_dist = distances_to_centres[i]
        closest = labels[i]
        for j in range(n_clusters):
            if not is_first_iteration and j == closest:
                continue
            bound = distances_between_centres[j, closest] / 2.0
            if min_dist < bound:
                continue

            dist = msm_distance(X[i], cluster_centres[j], window=window)
            if dist < min_dist:
                min_dist = dist
                closest = j

        labels[i] = closest
        distances_to_centres[i] = min_dist

    inertia = np.sum(distances_to_centres**2)
    if verbose:
        print(f"{inertia:.5f}", end=" --> ")
    return labels, distances_to_centres, inertia


# ==================== Assignment ====================


def _mean_average(X, window):
    cluster_centres = X.mean(axis=0)
    try:
        pw = msm_pairwise_distance(X, cluster_centres, window=window)
        distances_to_centres = pw.min(axis=1)
        return cluster_centres, distances_to_centres
    except:
        temp = ""


# ==================== Update ====================
def _kesba_update(
    X,
    cluster_centres,
    labels,
    n_clusters,
    random_state,
    window,
    iteration,
    distances_to_centres=None,
):

    for j in range(n_clusters):
        # cluster_centres[j], dist_to_centre = _mean_average(X[labels == j], window=window)
        cluster_centres[j], dist_to_centre = elastic_barycenter_average(
            X[labels == j],
            distance="msm",
            max_iters=50,
            random_state=random_state,
            return_distances=True,
            window=window,
            method="petitjean",
        )
        if distances_to_centres is not None:
            distances_to_centres[labels == j] = dist_to_centre

    if distances_to_centres is None:
        return cluster_centres

    return cluster_centres, distances_to_centres


# ==================== Update ====================


# ==================== Empty Cluster ====================
def _handle_empty_cluster(
    X: np.ndarray,
    cluster_centres: np.ndarray,
    distances_to_centres: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    window: float,
):
    empty_clusters = np.setdiff1d(np.arange(n_clusters), labels)
    j = 0
    if empty_clusters.size > 0:
        print("Handling empty cluster")

    while empty_clusters.size > 0:
        current_empty_cluster_index = empty_clusters[0]
        index_furthest_from_centre = distances_to_centres.argmax()
        cluster_centres[current_empty_cluster_index] = X[index_furthest_from_centre]
        curr_pw = msm_pairwise_distance(X, cluster_centres, window=window)
        labels = curr_pw.argmin(axis=1)
        distances_to_centres = curr_pw.min(axis=1)
        empty_clusters = np.setdiff1d(np.arange(n_clusters), labels)
        j += 1
        if j > n_clusters:
            raise EmptyClusterError

    return labels, cluster_centres, distances_to_centres


# ==================== Empty Cluster ====================


# ==================== Initialisation ====================
def _kesba_kmeans_plus_plus(
    X,
    random_state,
    n_clusters,
    window,
):
    initial_center_idx = random_state.randint(X.shape[0])
    indexes = [initial_center_idx]

    min_distances = msm_pairwise_distance(
        X, X[initial_center_idx], window=window
    ).flatten()
    labels = np.zeros(X.shape[0], dtype=int)

    for i in range(1, n_clusters):
        probabilities = min_distances / min_distances.sum()
        next_center_idx = random_state.choice(X.shape[0], p=probabilities)
        indexes.append(next_center_idx)

        new_distances = msm_pairwise_distance(
            X, X[next_center_idx], window=window
        ).flatten()

        closer_points = new_distances < min_distances
        min_distances[closer_points] = new_distances[closer_points]
        labels[closer_points] = i

    centers = X[indexes]
    return centers, min_distances, labels


def _first(X, n_clusters, window):
    first = X[0:n_clusters]
    pw = msm_pairwise_distance(X, first, window=window)
    labels = pw.argmin(axis=1)
    distances_to_centres = pw.min(axis=1)
    return first, distances_to_centres, labels


# ==================== Initialisation ====================
