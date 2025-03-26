"""."""

__maintainer__ = ["baraline"]
__all__ = ["LSHDTWClassifier"]

from typing import Union

import numpy as np

from aeon.classification.base import BaseClassifier
from aeon.similarity_search.collection.neighbors import GridIndexANN


class LSHDTWClassifier(BaseClassifier):
    """.

    Examples
    --------
    >>> from aeon.datasets import load_classification

    >>> X_train, y_train = load_classification("GunPoint", split="train")
    >>> X_test, y_test = load_classification("GunPoint", split="test")
    >>> g = LSHDTWClassifier(
        grid_deltas=[0.25, 0.35],
        k=3,
        K=3,
        normalize=True,
        n_jobs=-1,
        random_state=42,
    ).fit(X_train, y_train)
    >>> preds = g.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "X_inner_type": ["np-list", "numpy3D"],
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        k: int = 3,
        grid_deltas: float = 0.25,
        K: int = 3,
        normalize: bool = True,
        L_max: float = 1.0,
        L_min: int = 9,
        L_step: int = 0.05,
        n_jobs: int = 1,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> None:
        self.k = k
        self.grid_deltas = grid_deltas
        self.normalize = normalize
        self.K = K
        self.L_max = L_max
        self.L_min = L_min
        self.L_step = L_step
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(self, X, y):
        """Fit Classifier to training data.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        y: array-like or list
            The class labels for samples in X.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_".
        """
        self._indexer = GridIndexANN(
            self.grid_deltas,
            self.K,
            L_max=self.L_max,
            L_min=self.L_min,
            L_step=self.L_step,
            normalize=self.normalize,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        ).fit(X)
        self.y_ = y
        classes = np.unique(y)
        self.n_classes_ = len(classes)
        self._class_dict = {i: classes[i] for i in range(len(classes))}
        self._class_dict_rev = {classes[i]: i for i in range(len(classes))}
        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        return [self._class_dict[i] for i in np.argmax(self._predict_proba(X), axis=1)]

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        top_k, top_k_dist = self._indexer.predict(X, k=self.k)
        classes = np.zeros((len(X), self.n_classes_))
        for i in range(len(X)):
            for j in self.y_[top_k[i]]:
                classes[i, self._class_dict_rev[j]] += 1.0
            classes[i] /= classes[i].sum()
        return classes
