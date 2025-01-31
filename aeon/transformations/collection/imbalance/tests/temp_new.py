import os

import numpy as np


def check_first_n_are_equal(
    first, second, first_labs, second_labs, validate_labels=True
):

    for i in range(len(first)):
        first_val = first[i]
        found = False
        found_index = -1
        for j in range(len(second)):
            if isinstance(first_val, np.ndarray):
                if np.array_equal(first_val, second[j]):
                    found = True
                    found_index = j
                    break
            else:
                if first_val == second[j]:
                    found = True
                    found_index = j
                    break
        assert found
        if validate_labels:
            assert first_labs[i] == second_labs[found_index]


if __name__ == "__main__":
    from aeon.datasets import load_from_ts_file

    PATH_TO_IMBALANCE_DATA = "/Users/chrisholder/Documents/Research/datasets/euclidean"
    ORIGINAL = "/Users/chrisholder/Documents/Research/datasets/imbalanced_9_1"
    for dir in os.listdir(PATH_TO_IMBALANCE_DATA):
        curr_path = f"{PATH_TO_IMBALANCE_DATA}/{dir}"
        print(f"==================== Processing {dir} ====================")
        for dataset in sorted(os.listdir(f"{curr_path}")):
            print("Processing", dataset)
            X_train, y_train = load_from_ts_file(
                f"{curr_path}/{dataset}/{dataset}_TRAIN.ts"
            )
            X_test, y_test = load_from_ts_file(
                f"{curr_path}/{dataset}/{dataset}_TEST.ts"
            )

            X_train_original, y_train_original = load_from_ts_file(
                f"{ORIGINAL}/{dataset}/{dataset}_TRAIN.ts"
            )
            X_test_original, y_test_original = load_from_ts_file(
                f"{ORIGINAL}/{dataset}/{dataset}_TEST.ts"
            )

            validate_labels = True
            if dataset == "ElectricDevices":
                validate_labels = False
            check_first_n_are_equal(
                X_train_original, X_train, y_train_original, y_train, validate_labels
            )
