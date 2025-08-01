# pylint: disable=invalid-name

from random import randrange
from typing import List, Tuple, Dict, Any, Callable
from dataclasses import dataclass

from ..evaluations import rmse_metric


@dataclass
class MultipleLinearRegressionModel:
    """
    Represents a multiple linear regression model.

    Attributes
    ----------
    intercept : float
        The bias term (b₀) in the regression equation.
    coefficients : List[float]
        A list of weights (b₁ to bₙ), one for each input feature.
    """
    intercept: float
    coefficients: List[float]


def predict(row: List[float], model: MultipleLinearRegressionModel) -> float:
    """
    Predict the target value for a single input row using a trained
    multiple linear regression model.

    Parameters
    ----------
    row : List[float]
        Input feature values.
    model : MultipleLinearRegressionModel
        The trained model with an intercept and list of feature coefficients.

    Returns
    -------
    float
        The predicted target value.
    """
    return model.intercept + sum(coef * x for coef, x in zip(model.coefficients, row))


def coefficients_sgd(
    train_X: List[List[float]],
    train_y: List[float],
    l_rate: float,
    n_epoch: int
) -> MultipleLinearRegressionModel:
    """
    Estimate a multiple linear regression model using stochastic gradient descent.

    Parameters
    ----------
    train_X : List[List[float]]
        The input features for training; each row is a list of feature values.
    train_y : List[float]
        The target values corresponding to each row in train_X.
    l_rate : float
        The learning rate for gradient descent.
    n_epoch : int
        The number of passes over the training dataset.

    Returns
    -------
    MultipleLinearRegressionModel
        A trained regression model with learned intercept and coefficients.
    """
    n_features = len(train_X[0])
    coef = [0.0 for _ in range(n_features + 1)]  # coef[0] is intercept

    for epoch in range(n_epoch):
        sum_error = 0.0

        for x_row, y_true in zip(train_X, train_y):
            model = MultipleLinearRegressionModel(
                intercept=coef[0],
                coefficients=coef[1:]
            )

            yhat = predict(x_row, model)

            error = yhat - y_true
            sum_error += error ** 2
            coef[0] -= l_rate * error

            for i, x in enumerate(x_row, start=1):
                coef[i] -= l_rate * error * x

        print(f">epoch={epoch}, lrate={l_rate:.3f}, error={sum_error:.3f}")

    return MultipleLinearRegressionModel(
        intercept=coef[0],
        coefficients=coef[1:]
    )


def predict_multiple_linear_regression(
    test: List[Tuple[float, ...]],
    model: MultipleLinearRegressionModel
) -> List[float]:
    """
    Predict target values using a trained multiple linear regression model.

    Parameters
    ----------
    test : List[Tuple[float, ...]]
        Test dataset where each item is a tuple of input features (x1, x2, ..., xn).
    model : MultipleLinearRegressionModel
        The trained model containing intercept and a list of feature coefficients.

    Returns
    -------
    List[float]
        Predicted values for each row in the test set.
    """
    return [predict(list(row), model) for row in test]


def dataset_minmax(dataset: Dict[str, Any]) -> Dict[str, tuple]:
    """
    Return min and max values for each column in the dataset.

    Parameters
    ----------
    dataset : Dict[str, Any]
        A dictionary where each key is a column name and the value is a list of numeric values.

    Returns
    -------
    Dict[str, tuple]
        A dictionary mapping each column to a (min, max) tuple.
    """
    return {k: (min(v), max(v)) for k, v in dataset.items()}


def normalize_dataset(
    dataset: Dict[str, Any],
    min_max_vals: Dict[str, Any]
) -> Dict[str, List[float]]:
    """
    Normalize each column in the dataset to the range [0, 1].

    Parameters
    ----------
    dataset : Dict[str, Any]
        A dictionary where each key is a column name and the value is a list of numeric values.
    min_max_vals : Dict[str, Any]
        A dictionary where each key is a column name and the value is a (min, max) tuple.

    Returns
    -------
    Dict[str, List[float]]
        A dictionary with normalized values for each column.
    """
    return {
        k: [
            (v - min_max_vals[k][0]) /
            (min_max_vals[k][1] - min_max_vals[k][0])
            if min_max_vals[k][1] != min_max_vals[k][0] else 0.0
            for v in values
        ]
        for k, values in dataset.items()
    }


def cross_validation_split(
    X: List[float],
    Y: List[float],
    n_folds: int
) -> List[List[Tuple[float, float]]]:
    """
    Randomly split X and Y into k folds for cross-validation.

    Parameters
    ----------
    X : List[float]
        Input features.
    Y : List[float]
        Target values (must match X in length).
    n_folds : int
        Number of folds to split the data into.

    Returns
    -------
    List[List[Tuple[float, float]]]
        A list of folds, each containing (x, y) tuples.
    """
    assert len(X) == len(Y), "X and Y must be the same length"

    dataset = list(zip(X, Y))
    dataset_copy = dataset.copy()

    fold_size = int(len(dataset) / n_folds)
    dataset_split = []

    for _ in range(n_folds):
        fold = []

        while len(fold) < fold_size and dataset_copy:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))

        dataset_split.append(fold)

    return dataset_split


def linear_regression_sgd(
    train: List[Tuple[float, float]],
    l_rate: float = 0.001,
    n_epoch: int = 5
) -> MultipleLinearRegressionModel:
    """
    Train a multiple linear regression model using SGD.

    Parameters
    ----------
    train : List[Tuple[float, float]]
        Training data as (x, y) pairs.
    l_rate : float, optional
        Learning rate for SGD. Defaults to 0.001
    n_epoch : int, optional
        Number of training epochs. Defaults to 5.

    Returns
    -------
    MultipleLinearRegressionModel
        A trained regression model with learned intercept and coefficients.
    """
    train_X = [[x] if not isinstance(
        x, (list, tuple)) else list(x) for x, _ in train]
    train_y = [y for _, y in train]

    return coefficients_sgd(train_X, train_y, l_rate, n_epoch)


def evaluate_mlr(
    X: List[float],
    Y: List[float],
    train_func: Callable[[List[Tuple[float, float]]], MultipleLinearRegressionModel],
    predict_func: Callable[[List[Tuple[float]], MultipleLinearRegressionModel], List[float]],
    n_folds: int,
    l_rate: float = 0.001,
    n_epoch: int = 50
) -> List[float]:
    """
    Evaluate a regression algorithm using k-fold cross-validation.

    Parameters
    ----------
    X : List[float]
        Input feature values.
    Y : List[float]
        Target values.
    train_func : Callable
        Function that trains a model and returns a model object.
    predict_func : Callable
        Function that accepts test input and model, and returns predictions.
    n_folds : int
        Number of folds for cross-validation.

    Returns
    -------
    List[float]
        RMSE score for each fold.
    """
    folds = cross_validation_split(X, Y, n_folds)
    scores = []

    for i, fold in enumerate(folds):
        test_set = fold
        train_set = [row for j, f in enumerate(folds) if j != i for row in f]

        x_test = [list(x) for x, _ in test_set]
        y_actual = [y for _, y in test_set]

        model = train_func(train_set, l_rate=l_rate, n_epoch=n_epoch)
        y_predicted = predict_func(x_test, model)

        rmse = rmse_metric(y_actual, y_predicted)
        scores.append(rmse)

    return scores
