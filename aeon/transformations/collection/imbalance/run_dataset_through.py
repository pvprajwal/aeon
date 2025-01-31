import os
from pathlib import Path

import numpy as np

from aeon.datasets import load_from_ts_file
from aeon.transformations.collection.imbalance import ADASYN, SMOTE, TSMOTE

PATH_TO_IMBALANCE_DATA = "/Users/chrisholder/Documents/Research/datasets/imbalanced_9_1"


def save_to_ts_file(X, y, output_path, input_path):
    """Save data to .ts file format, preserving original headers.

    Parameters
    ----------
    X : np.ndarray
        Time series data
    y : np.ndarray
        Target labels
    output_path : str
        Path to save the .ts file
    input_path : str
        Path to original .ts file to copy headers from
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read original headers
    headers = []
    with open(input_path) as f:
        for line in f:
            if line.startswith("@"):
                headers.append(line)
            elif line.startswith("@data"):
                headers.append(line)
                break

    # Write file with original headers and new data
    with open(output_path, "w") as f:
        # Write original headers
        for header in headers:
            f.write(header)

        # Write data
        for i in range(len(X)):
            # Write target value
            # Write time series values
            if len(X.shape) == 3:  # For 3D array (n_samples, n_channels, n_timepoints)
                f.write(",".join(str(val) for val in X[i, 0, :]))
            else:  # For 2D array (n_samples, n_timepoints)
                f.write(",".join(str(val) for val in X[i, :]))
            f.write(":" + str(y[i]))
            f.write("\n")


def process_dataset(input_path, output_base_path, transformer, dataset_name, split):
    """Process a single dataset file with the given transformer.

    Parameters
    ----------
    input_path : str
        Path to input .ts file
    output_base_path : str
        Base path for output directory
    transformer : object
        Rebalancing transformer (SMOTE, ADASYN, etc.)
    dataset_name : str
        Name of the dataset
    split : str
        Either 'TRAIN' or 'TEST'
    """
    # Load data
    X, y = load_from_ts_file(input_path)

    # Apply transformation
    if split == "TRAIN":  # Only transform training data
        X_res, y_res = transformer.fit_transform(X, y)
    else:
        X_res, y_res = X, y  # Don't transform test data

    # Create output path
    output_path = os.path.join(
        output_base_path, dataset_name, f"{dataset_name}_{split}.ts"
    )

    # Save transformed data
    save_to_ts_file(X_res, y_res, output_path, input_path)


if __name__ == "__main__":
    distance = "dtw"

    if distance == "dtw":
        distance_params = {"window": 0.2}
    elif distance == "msm":
        distance_params = {"c": 1, "independent": True}
    else:
        distance_params = {}

    # Define transformers
    transformers = {
        "smote": SMOTE(distance=distance, distance_params=distance_params),
        "adasyn": ADASYN(distance=distance, distance_params=distance_params),
        "tsmote": TSMOTE(distance=distance, distance_params=distance_params),
    }

    # Process each transformer
    for transformer_name, transformer in transformers.items():
        # Create output directory for this transformer
        print(f"Processing {transformer_name}...")
        output_base_path = f"{PATH_TO_IMBALANCE_DATA}_{transformer_name}"

        # Iterate through all datasets in the input directory
        for dataset_name in sorted(os.listdir(PATH_TO_IMBALANCE_DATA)):
            dataset_path = os.path.join(PATH_TO_IMBALANCE_DATA, dataset_name)

            # Skip if not a directory
            if not os.path.isdir(dataset_path):
                continue

            print(f"Processing {dataset_name} with {transformer_name}...")

            # Process both train and test splits
            for split in ["TRAIN", "TEST"]:
                input_path = os.path.join(dataset_path, f"{dataset_name}_{split}.ts")

                if os.path.exists(input_path):
                    try:
                        process_dataset(
                            input_path,
                            output_base_path,
                            transformer,
                            dataset_name,
                            split,
                        )
                    except Exception as e:
                        print(f"Error processing {input_path}: {str(e)}")

            X_train, y_train = load_from_ts_file(
                f"{output_base_path}/{dataset_name}/{dataset_name}_TRAIN.ts"
            )
            X_test, y_test = load_from_ts_file(
                f"{output_base_path}/{dataset_name}/{dataset_name}_TEST.ts"
            )
