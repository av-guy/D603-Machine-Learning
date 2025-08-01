"""
distance_metrics.py

Distance function calculations for numerical vectors.

All functions assume input vectors are real-valued and of equal length.

Functions
---------
euclidean_distance(j, k)
    Computes the Euclidean (L2) distance between two vectors.

manhattan_distance(j, k)
    Computes the Manhattan (L1) distance between two vectors.

minkowski_distance(j, k, p)
    Computes the Minkowski distance of order `p` between two vectors.
"""


from typing import List


def euclidean_distance(j: List[float], k: List[float]) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Parameters
    ----------
    j : List[float]
        First vector.
    k : List[float]
        Second vector.

    Returns
    -------
    float
        Euclidean distance between vectors j and k.

    Raises
    ------
    AssertionError
        If the input vectors are not of the same length.
    """
    assert len(j) == len(k)
    return sum((xi_j - xi_k) ** 2 for xi_j, xi_k in zip(j, k)) ** 0.5


def manhattan_distance(j: List[float], k: List[float]) -> float:
    """
    Compute the Manhattan (L1) distance between two vectors.

    Parameters
    ----------
    j : List[float]
        First vector.
    k : List[float]
        Second vector.

    Returns
    -------
    float
        Manhattan distance between vectors j and k.

    Raises
    ------
    AssertionError
        If the input vectors are not of the same length.
    """
    assert len(j) == len(k)
    return sum(abs(xi_j - xi_k) for xi_j, xi_k in zip(j, k))


def minkowski_distance(j: List[float], k: List[float], p: float) -> float:
    """
    Compute the Minkowski distance of order `p` between two vectors.

    Parameters
    ----------
    j : List[float]
        First vector.
    k : List[float]
        Second vector.
    p : float
        Order of the norm (e.g., 1 for Manhattan, 2 for Euclidean).

    Returns
    -------
    float
        Minkowski distance of order `p` between vectors j and k.

    Raises
    ------
    AssertionError
        If the input vectors are not of the same length.
    """
    assert len(j) == len(k)

    if p == 1:
        return manhattan_distance(j, k)

    return sum(abs(xi_j - xi_k) ** p for xi_j, xi_k in zip(j, k)) ** (1 / p)
