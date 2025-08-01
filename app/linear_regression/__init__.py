from typing import Tuple, List
from random import seed, randrange

from .simple_regression import (
    SimpleLinearRegressionModel,
    train_simple_linear_regression,
    predict_simple_linear_regression,
    evaluate_model
)

from .multiple_regression import (
    MultipleLinearRegressionModel,
    coefficients_sgd,
    predict,
    predict_multiple_linear_regression,
    linear_regression_sgd,
    normalize_dataset,
    dataset_minmax,
    cross_validation_split,
    evaluate_mlr as evaluate_mlr
)


def train_test_split(
    x: List[float],
    y: List[float],
    split: float
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Randomly split X and Y into train and test datasets based on a split ratio.

    Parameters
    ----------
    x : List[float]
        List of input features.
    y : List[float]
        List of target values (must be the same length as X).
    split : float
        Proportion of data to use for training (e.g., 0.6 means 60% train, 40% test).

    Returns
    -------
    Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]
        Two lists of (x, y) tuples: the training set and the test set.
    """
    assert len(x) == len(y), "X and Y must be the same length"

    dataset = list(zip(x, y))
    train_size = int(split * len(dataset))
    dataset_copy = dataset.copy()
    train_set = []

    while len(train_set) < train_size:
        index = randrange(len(dataset_copy))
        train_set.append(dataset_copy.pop(index))

    test_set = dataset_copy
    return train_set, test_set
