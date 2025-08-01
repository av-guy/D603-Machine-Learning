from math import sqrt
from typing import List


def rmse_metric(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate root mean squared error (RMSE) between actual and predicted values.

    Parameters
    ----------
    actual : List[float]
        The ground truth target values.
    predicted : List[float]
        The predicted target values.

    Returns
    -------
    float
        The RMSE between the actual and predicted values.
    """
    sum_error = sum((pred - act) ** 2 for act, pred in zip(actual, predicted))
    return sqrt(sum_error / len(actual))
