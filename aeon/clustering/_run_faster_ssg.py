import numpy as np


def _get_distance_default_params(dist_name: str) -> dict:
    if dist_name == "dtw" or dist_name == "ddtw":
        return {"window": 0.2}
    if dist_name == "lcss":
        return {"epsilon": 1.0}
    if dist_name == "msm":
        return {"c": 1.0, "independent": True}
    if dist_name == "edr":
        return {"epsilon": None}
    if dist_name == "twe":
        return {"nu": 0.001, "lmbda": 1.0}
    if dist_name == "psi_dtw":
        return {"r": 0.5}
    if dist_name == "adtw":
        return {"warp_penalty": 1.0}
    if dist_name == "shape_dtw":
        return {"descriptor": "identity", "reach": 30}
    return {}


def _set_experimental_clusterer(
    c,
    random_state,
    n_clusters,
    kwargs,
):
    distance = c.split("-")[-1]
    distance_params = _get_distance_default_params(distance)

    if "window" in c:
        distance_params = {**distance_params, "window": 0.2}
    average_params = {"distance": distance, **distance_params.copy()}

    if "faster" in c:
        average_params = {
            **average_params,
            "method": "holdit",
        }
    elif "proper-stopping" in c:
        average_params = {**average_params, "method": "holdit_stopping"}
    elif "approx-stopping" in c:
        average_params = {
            **average_params,
            "method": "holdit_stopping_approx",
        }
    elif "avg-change-stopping" in c:
        average_params = {
            **average_params,
            "method": "holdit_stopping_avg_change",
        }
    else:
        average_params = {
            **average_params,
            "method": "subgradient",
        }

    potential_size_arg = ["50", "40", "30", "20", "10"]
    if any(arg in c for arg in potential_size_arg):
        size = int(c.split("-")[0])
        average_params = {
            **average_params,
            "holdit_num_ts_to_use_percentage": size / 100,
        }
    from aeon.clustering._holdit_k_means import HoldItKmeans

    return HoldItKmeans(
        n_clusters=n_clusters,
        max_iter=300,
        n_init=1,
        init_algorithm="kmeans++",
        distance=distance,
        distance_params=distance_params,
        random_state=random_state,
        averaging_method="ba",
        average_params={
            **average_params,
        },
        verbose=True,
        **kwargs,
    )

    # return TimeSeriesKMeans(
    #     n_clusters=n_clusters,
    #     max_iter=50,
    #     n_init=10,
    #     init_algorithm="random",
    #     distance=distance,
    #     distance_params=distance_params,
    #     random_state=random_state,
    #     averaging_method="mean",
    #     average_params=average_params,
    #     verbose=True,
    #     **kwargs,
    # )


if __name__ == "__main__":
    from aeon.clustering import TimeSeriesKMeans
    from aeon.datasets import load_from_tsfile as load_tsfile

    DATA_PATH = "/home/chris/Documents/Univariate_ts"

    CLUSTERER = "proper-stopping-ssg-adtw"
    DATASET_NAME = "ACSF1"

    X_train, y_train = load_tsfile(
        f"{DATA_PATH}/{DATASET_NAME}/{DATASET_NAME}_TRAIN.ts"
    )
    X_test, y_test = load_tsfile(f"{DATA_PATH}/{DATASET_NAME}/{DATASET_NAME}_TEST.ts")

    X_train = X_train[:, :, :100]
    y_train = y_train[:100]
    X_test = X_test[:, :, :100]
    y_test = y_test[:100]

    clusterer = _set_experimental_clusterer(
        c=CLUSTERER, n_clusters=len(np.unique(y_train)), random_state=1, kwargs={}
    )

    clusterer.fit(X_train)
    print(clusterer.labels_)

    temp = ""
