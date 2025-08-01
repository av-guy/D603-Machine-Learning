
""" Simple linear regression methods """

from typing import List, Tuple, Callable
from dataclasses import dataclass

from ..stats import coefficients
from ..evaluations import rmse_metric


@dataclass
class SimpleLinearRegressionModel:
    """
    A simple data class to store the coefficients of a linear regression model.

    Attributes
    ----------
    intercept : float
        The intercept (b0) of the regression line.
    slope : float
        The slope (b1) of the regression line.
    """
    intercept: float
    slope: float


def train_simple_linear_regression(train: List[Tuple[float, float]]) -> SimpleLinearRegressionModel:
    """
    Fit a simple linear regression model to the training data.

    Parameters
    ----------
    train : List[Tuple[float, float]]
        Training dataset where each item is a tuple (x, y).

    Returns
    -------
    SimpleLinearRegressionModel
        The fitted model with learned intercept and slope.
    """
    x = [row[0] for row in train]
    y = [row[1] for row in train]

    b0, b1 = coefficients(x, y)
    return SimpleLinearRegressionModel(intercept=b0, slope=b1)


def predict_simple_linear_regression(
    test: List[Tuple[float]],
    model: SimpleLinearRegressionModel
) -> List[float]:
    """
    Predict target values using a trained simple linear regression model.

    Parameters
    ----------
    test : List[Tuple[float]]
        Test dataset where each item is a tuple containing a single input value (x,).
    model : SimpleLinearRegressionModel
        The trained model containing intercept and slope.

    Returns
    -------
    list of float
        Predicted values for each input x in the test set.
    """
    return [model.intercept + model.slope * row[0] for row in test]


def evaluate_model(
    train: List[Tuple[float, float]],
    test: List[Tuple[float, float]],
    train_func: Callable[[List[Tuple[float, float]]], SimpleLinearRegressionModel],
    predict_func: Callable[[List[Tuple[float]],
                            SimpleLinearRegressionModel], List[float]]
) -> float:
    """
    Evaluate a regression model on a test set using RMSE.

    Parameters
    ----------
    train : List[Tuple[float, float]]
        The training dataset where each item is a tuple (x, y).
    test : List[Tuple[float, float]]
        The test dataset where each item is a tuple (x, y).
    train_func : Callable[[List[Tuple[float, float]]], SimpleLinearRegressionModel]
        A function that trains a model from the training dataset and returns a model instance.
    predict_func : Callable[[List[Tuple[float]],
                            SimpleLinearRegressionModel], List[float]]
        A function that accepts test inputs and a trained model, and returns predictions.

    Returns
    -------
    float
        The root mean squared error (RMSE) between predicted and actual y values in the test set.
    """
    x_test = [(row[0],) for row in test]
    y_actual = [row[1] for row in test]

    model = train_func(train)
    y_predicted = predict_func(x_test, model)

    return rmse_metric(y_actual, y_predicted)
