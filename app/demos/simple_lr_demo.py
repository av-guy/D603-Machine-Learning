from typing import List, Tuple
from random import seed, randrange

from ..linear_regression import (
    evaluate_model,
    train_simple_linear_regression,
    predict_simple_linear_regression
)

from ..files import load_csv

seed(1)

insurance = load_csv("data", "insurance.csv", {"X": lambda x: float(
    x.strip()), "Y": lambda x: float(x.strip())})


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


def main():
    """ Run the demo """
    train, test = train_test_split(insurance['X'], insurance['Y'], split=0.6)

    rmse = evaluate_model(
        train,
        test,
        train_func=train_simple_linear_regression,
        predict_func=predict_simple_linear_regression
    )

    print(f"RMSE: {rmse:.3f}")
