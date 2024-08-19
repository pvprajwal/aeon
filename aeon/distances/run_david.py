from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from aeon.datasets import load_from_tsfile
from aeon.distances import shape_dtw_pairwise_distance
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.testing.data_generation import make_example_3d_numpy

if __name__ == "__main__":

    X_train_mtser, y_train_mtser = make_example_3d_numpy(
        n_cases=72, n_channels=4, n_timepoints=100, return_y=True
    )
    X_test_mtser, y_test_mtser = make_example_3d_numpy(
        n_cases=72, n_channels=4, n_timepoints=100, return_y=True
    )
    # X_train_mtser, y_train_mtser = load_from_tsfile("/Users/chris/projects/aeon-tutorials/KDD-2024/notebooks/data/KDD_MTSER_TRAIN.ts")
    # X_test_mtser, y_test_mtser = load_from_tsfile("/Users/chris/projects/aeon-tutorials/KDD-2024/notebooks/data/KDD_MTSER_TEST.ts")

    print("Train shape:", X_train_mtser.shape)
    print("Test shape:", X_test_mtser.shape)

    # Now, for the shape_dtw distance:
    train_dists = shape_dtw_pairwise_distance(X_train_mtser, window=0.1)
    test_dists = shape_dtw_pairwise_distance(X_test_mtser, X_train_mtser, window=0.1)
    knn = KNeighborsRegressor(n_neighbors=1, metric="precomputed")
    knn.fit(train_dists, y_train_mtser)
    predictions_mtser = knn.predict(test_dists)

    mse = mean_squared_error(y_test_mtser, predictions_mtser)
    print(f"MSE of 1-NN (sklearn) with shape_dtw is {mse}")

    # TODO: modify knnregressor to work with callables.
    knn_aeon = KNeighborsTimeSeriesRegressor(
        n_neighbors=1,
        distance="shape_dtw",
        distance_params={
            "window": 0.1,
        },
    )
    knn_aeon.fit(X_train_mtser, y_train_mtser)
    predictions_mtser_aeon = knn_aeon.predict(X_test_mtser)

    mse = mean_squared_error(y_test_mtser, predictions_mtser_aeon)
    print(f"MSE of 1-NN (aeon) with shape_dtw is {mse}")
