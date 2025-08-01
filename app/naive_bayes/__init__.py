# pylint: disable=invalid-name

from typing import List
from math import pi, sqrt, exp


def bayes_rule(p_A: float, p_B: float, p_BA: float) -> float:
    """
    Applies Bayes' Theorem to compute the conditional probability P(A|B).

    Parameters
    ----------
    p_A : float
        The prior probability of event A, P(A).
    p_B : float
        The total probability of event B, P(B).
    p_BA : float
        The conditional probability of event B given A, P(B|A).

    Returns
    -------
    float
        The conditional probability of event A given B, P(A|B).

    Raises
    ------
    ZeroDivisionError
        If p_B is zero, as division by zero is undefined.
    """
    return p_BA * p_A / p_B


def calculate_normal_dist(mu: float, sigma: float, x: float) -> float:
    """
    Calculates the probability density of the normal distribution at a given point.

    Parameters
    ----------
    mu : float
        The mean (μ) of the normal distribution.
    sigma : float
        The standard deviation (σ) of the normal distribution. Must be positive.
    x : float
        The point at which to evaluate the probability density function (PDF).

    Returns
    -------
    float
        The value of the normal distribution's PDF at the given point x.

    Raises
    ------
    ZeroDivisionError
        If sigma is zero, as division by zero is undefined.
    ValueError
        If sigma is negative, since standard deviation must be positive.

    Notes
    -----
    This computes the probability density function (PDF) of a normal (Gaussian)
    distribution:

        f(x) = (1 / (√(2πσ²))) * exp(-((x - μ)²) / (2σ²))
    """
    if sigma <= 0:
        raise ValueError("Standard deviation sigma must be positive.")

    return (1 / (sqrt(2 * pi) * sigma)) * exp(-((x - mu) ** 2) / (2 * sigma ** 2))


if __name__ == "__main__":
    pA = 39 / 500.0
    pB = 39.0 / 500.0
    pBA = 27 / 39

    pAB = bayes_rule(pA, pB, pBA)

    s_x = 200
    mu_hd = 177.5
    sigma_hd = 20.05

    mu_nhd = 140.3
    sigma_nhd = 23.44

    x_hd = calculate_normal_dist(mu_hd, sigma_hd, 200)
    x_nhd = calculate_normal_dist(mu_nhd, sigma_nhd, 200)

    print(x_hd, x_nhd)
