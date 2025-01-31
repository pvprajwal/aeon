import numpy as np

if __name__ == "__main__":
    from aeon.datasets import load_from_ts_file

    DATA_PATH = "/Users/chrisholder/Documents/Research/datasets/UCR/Univariate_ts"

    dataset = "ElectricDevices"

    X_train, y_train = load_from_ts_file(f"{DATA_PATH}/{dataset}/{dataset}_TEST.ts")
    total_duplicates = 0
    total_duplicates_with_diff_labels = 0

    for i in range(len(X_train)):
        first = X_train[i]
        for j in range(len(X_train)):
            second = X_train[j]

            if j != i and np.array_equal(first, second):
                total_duplicates += 1
                if y_train[i] != y_train[j]:
                    total_duplicates_with_diff_labels += 1
                    temp1 = first
                    temp2 = second
                    equal = np.array_equal(temp1, temp2)
                    temp1_label = y_train[i]
                    temp2_label = y_train[j]
                    print(
                        f"Found duplicate at index {i} and {j} with different class labels. First label: {temp1_label}, Second label: {temp2_label}"
                    )

    print(f"Total duplicates: {total_duplicates}")
    print(
        f"Total duplicates with different class labels: {total_duplicates_with_diff_labels}"
    )
