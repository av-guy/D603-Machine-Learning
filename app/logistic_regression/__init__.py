# pylint: disable=invalid-name

""" Logistic Regression Module """

from math import exp
from typing import List, Tuple, Callable
from dataclasses import dataclass

from ..linear_regression import cross_validation_split


@dataclass
class LogisticRegressionModel:
    """
    Logistic regression model parameters.

    Attributes
    ----------
    intercept : float
        The bias term (intercept) of the model.
    coefficients : List[float]
        Coefficients corresponding to input features.
    """
    intercept: float
    coefficients: List[float]


def predict(row: List[float], model: LogisticRegressionModel) -> float:
    """
    Predict the probability using a trained logistic regression model.

    Parameters
    ----------
    row : List[float]
        Input feature values.
    model : LogisticRegressionModel
        The trained model with an intercept and list of feature coefficients.

    Returns
    -------
    float
        The predicted probability between 0 and 1.
    """
    linear_sum = model.intercept + \
        sum(coef * x for coef, x in zip(model.coefficients, row))

    return 1.0 / (1.0 + exp(-linear_sum))


def logistic_regression_sgd(
    train: List[Tuple[float, float]],
    l_rate: float = 0.001,
    n_epoch: int = 5
) -> LogisticRegressionModel:
    """
    Train a logistic regression model using stochastic gradient descent.

    Parameters
    ----------
    train : List[Tuple[float, float]]
        Training data as (X, y) pairs.
    l_rate : float, optional
        Learning rate (default is 0.001).
    n_epoch : int, optional
        Number of training epochs (default is 5).

    Returns
    -------
    LogisticRegressionModel
        Trained model with learned intercept and coefficients.
    """
    train_X = [list(x) if isinstance(x, (list, tuple)) else [x]
               for x, _ in train]
    train_y = [y for _, y in train]

    return coefficients_sgd(train_X, train_y, l_rate, n_epoch)


def predict_logistic_regression(
    test: List[List[float]],
    model: LogisticRegressionModel
) -> List[int]:
    """
    Predict binary class labels using a trained logistic regression model.

    Parameters
    ----------
    test : List[List[float]]
        List of input feature vectors.
    model : LogisticRegressionModel
        Trained logistic regression model.

    Returns
    -------
    List[int]
        Predicted class labels (0 or 1) for each input.
    """
    return [round(predict(row, model)) for row in test]


def coefficients_sgd(
    train_X: List[List[float]],
    train_y: List[float],
    l_rate: float,
    n_epoch: int
) -> LogisticRegressionModel:
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
            model = LogisticRegressionModel(
                intercept=coef[0],
                coefficients=coef[1:]
            )

            yhat = predict(x_row, model)

            error = y_true - yhat
            sum_error += error ** 2
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)

            for i, x in enumerate(x_row, start=1):
                coef[i] = coef[i] + l_rate * error * yhat * (1.0 - yhat) * x

        print(f">epoch={epoch}, lrate={l_rate:.3f}, error={sum_error:.3f}")

    return LogisticRegressionModel(
        intercept=coef[0],
        coefficients=coef[1:]
    )


def accuracy_metric(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate the classification accuracy as a percentage.

    Parameters
    ----------
    actual : List[float]
        The true class labels.
    predicted : List[float]
        The predicted class labels.

    Returns
    -------
    float
        The accuracy as a percentage of correct predictions.
    """
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual) * 100.0


def evaluate_log(
    X: List[float],
    Y: List[float],
    train_func: Callable[[List[Tuple[float, float]]], LogisticRegressionModel],
    predict_func: Callable[[List[Tuple[float]], LogisticRegressionModel], List[float]],
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

        acc = accuracy_metric(y_actual, y_predicted)
        scores.append(acc)

    return scores
