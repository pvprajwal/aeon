""".

import numpy as np
import optuna
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


def _get_scores_sample(X_buck, n_timestamps, max_ts):
    presence = 0
    occurences = 0
    loc_distrib = np.zeros(max_ts)
    buck_size = 0
    mask = np.ones(n_timestamps, dtype=bool)
    for coords in X_buck:
        buck_size += 1
        loc_distrib[coords[1] : coords[1] + coords[2]] += 1
        mask_size = int(coords[2] * mask_percentage)
        if mask[coords[1]]:
            mask[coords[1] - mask_size : coords[1] + mask_size] = False
            occurences += 1
            presence = 1

    return presence, loc_distrib / buck_size, occurences


def _get_scores(buck, y, n_timestamps, class_dict, mask_percentage=0.25):
    if len(buck) == 1:
        return None
    else:
        n_classes = len(class_dict)
        max_ts = max(n_timestamps)
        presence_score = np.zeros(n_classes)
        location_score = np.zeros((n_classes, max_ts))
        occurence_score = np.zeros(n_classes)
        occurence_count = np.zeros(n_classes)

        i_samples = np.unique(buck[:, 0])
        for i_sample in i_samples:
            presence = 0
            occurences = 0
            loc_distrib = np.zeros(max_ts)
            buck_size = 0
            mask = np.ones(n_timestamps[i_sample], dtype=bool)
            for coords in buck[buck[:, 0] == i_sample]:
                buck_size += 1
                loc_distrib[coords[1] : coords[1] + coords[2]] += 1
                mask_size = int(coords[2] * mask_percentage)
                if mask[coords[1]]:
                    mask[coords[1] - mask_size : coords[1] + mask_size] = False
                    occurences += 1
                    presence = 1

            presence_score[class_dict[y[i_sample]]] += presence

            location_score[class_dict[y[i_sample]]] += loc_distrib / buck_size

            occurence_count[class_dict[y[i_sample]]] += 1
            occurence_score[class_dict[y[i_sample]]] += occurences

        occurence_score[occurence_count > 0] /= occurence_count[occurence_count > 0]

        return presence_score, location_score, occurence_score


def _trial(
    X,
    y,
    trial,
    random_state=42,
    n_splits=5,
):
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state, test_size=1 / n_splits
    )
    scores = []
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        g = GridIndexANN(
            trial.params["grid_delta"],
            trial.params["K"],
            n_hash_funcs=trial.params["n_hash_funcs"],
            random_state=random_state,
            normalize=trial.params["K"],
            n_jobs=1,
        )
        g.fit(X_train, y_train)
        y_pred = g.predict(X_test, method=trial.params["pred_mode"])
        scores.append(accuracy_score(y_test, y_pred))
        return np.mean(scores)


def _optimize(X, y, n_splits=3, random_state=42, n_trials=100):
    study = optuna.create_study(direction="maximize")
    distributions = {
        "grid_delta": optuna.distributions.FloatDistribution(0.05, 1),
        "n_hash_funcs": optuna.distributions.IntDistribution(1, 8),
        "K": optuna.distributions.IntDistribution(1, 5),
        "normalize": optuna.distributions.CategoricalDistribution([True, False]),
        "pred_mode": optuna.distributions.CategoricalDistribution(
            ["presence_rep", "occurence_rep"]
        ),
    }
    for i in range(n_trials):
        trial = study.ask(distributions)
        score = _trial(X, y, trial, random_state=random_state, n_splits=n_splits)
        print(f"trial {i}/{n_trials} score : {score}")
        study.tell(trial, score)
    return study.best_params
"""
