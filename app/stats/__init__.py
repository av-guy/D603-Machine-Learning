"""
Basic statistical functions for numerical data.

This module provides simple implementations of fundamental statistical
operations: arithmetic mean and sample standard deviation. It supports
both lists and tuples of numeric values (integers or floats).

Functions
---------
mean(x)
    Computes the arithmetic mean of the input values.

std(x, m = None)
    Computes the sample standard deviation of the input values using Bessel's correction.
"""

from typing import List, Tuple, Optional


def mean(x: List[float | int] | Tuple[float | int, ...]) -> float:
    """
    Calculate the arithmetic mean of the input values.

    Parameters
    ----------
    x : List[float | int] | Tuple[float | int, ...]
        Numeric values to compute the mean.

    Returns
    -------
    float
        The arithmetic mean of the input values.
    """
    return sum(x) / float(len(x))


def std(x: List[float | int] | Tuple[float | int, ...], m: Optional[float] = None) -> float:
    """
    Calculate the sample standard deviation of the input values.

    Parameters
    ----------
    x : List[float | int] | Tuple[float | int, ...]
        Numeric values to compute the standard deviation.
    m : float, optional
        The mean value. If not provided, mean will be calculated using values.

    Returns
    -------
    float
        The sample standard deviation of the input values.
    """
    m = mean(x) if m is None else m
    squared_diffs = [(x - m) ** 2 for x in x]

    return (sum(squared_diffs) / (len(x) - 1)) ** 0.5


def variance(x: List[float | int], m: Optional[float] = None) -> float:
    """
    Calculate the sum of squared deviations from the mean (unnormalized variance).

    Parameters
    ----------
    x : List[float | int]
        The data values.
    m : float
        The mean of the data.

    Returns
    -------
    float
        The sum of squared differences between each value and the mean.
    """
    m = mean(x) if m is None else m
    return sum(((x - m) ** 2 for x in x))


def covariance(
    x: List[float | int] | Tuple[float | int, ...],
    y: List[float | int] | Tuple[float | int, ...],
    mean_x: float,
    mean_y: float
) -> float:
    """
    Calculate the covariance between two numeric sequences given their means.

    Parameters
    ----------
    x : List[float | int] | Tuple[float | int, ...]
        The first sequence of numerical values.
    y : List[float | int] | Tuple[float | int, ...]
        The second sequence of numerical values, of the same length as `x`.
    mean_x : float
        The mean of the values in `x`.
    mean_y : float
        The mean of the values in `y`.

    Returns
    -------
    float
        The sum of the product of deviations from the mean for each pair (x_i, y_i).
    """
    return float(sum(((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))))


def coefficients(x: List[float | int], y: List[float | int]) -> Tuple[float, float]:
    """
    Calculate the coefficients for simple linear regression using least squares.

    Parameters
    ----------
    x : List[float | int]
        The independent variable values.
    y : List[float | int]
        The dependent variable values.

    Returns
    -------
    tuple of float
        A tuple (b0, b1) where:
        - b0 is the intercept of the regression line
        - b1 is the slope of the regression line
    """
    x_mean, y_mean = mean(x), mean(y)

    b1 = covariance(x, y, x_mean, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean

    return b0, b1
