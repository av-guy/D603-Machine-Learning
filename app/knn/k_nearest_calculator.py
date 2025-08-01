"""
knn_classifier.py

Simple implementation of the k-nearest neighbors (KNN) classification algorithm.

This module provides a general-purpose function for predicting the class label of a 
numeric input vector (`x_star`) based on labeled training data grouped by identifiers. 
It supports multiple distance metrics (Manhattan, Euclidean, Minkowski), and assumes 
that the input feature values are real numbers.

Typical use case includes classifying a new observation by comparing it to 
previously observed and labeled data using a selected distance metric and majority vote.

Functions
---------
k_nearest_neigbhors(x_star, identifiers_dict, k=5, metric='manhattan')
    Classify a numeric input vector using KNN and return the predicted label.
"""

from typing import Dict, List, Tuple, Union
from collections import Counter
from . import euclidean_distance, manhattan_distance, minkowski_distance


METRICS = {
    'manhattan': manhattan_distance,
    'euclidean': euclidean_distance,
    'minkowski': minkowski_distance
}


def k_nearest_neigbhors(
    x_star: Union[List[float], Tuple[float, ...]],
    identifiers_dict: Dict[str, Dict[str, List[float]]],
    k: int = 5,
    metric: str = 'manhattan'
) -> str:
    """
    Predict the identifier for a given input using k-nearest neighbors.

    Parameters
    ----------
    x_star : Union[List[float], Tuple[float, ...]]
        The input vector to classify.
    identifiers_dict : Dict[str, Dict[str, List[float]]]
        Dictionary mapping identifiers to feature names and their lists of values.
    k : int, optional
        Number of nearest neighbors to consider (default is 5).
    metric : str, optional
        Distance metric to use: 'manhattan', 'euclidean', or 'minkowski'.

    Returns
    -------
    str
        The predicted identifier/class label.
    """
    metric = metric.lower()

    if metric not in METRICS:
        raise ValueError(
            f"{metric} is not a valid metric, use one of {', '.join(METRICS.keys())}")

    dist_metric = METRICS[metric]
    all_distances = []

    for identifier, feature_dict in identifiers_dict.items():
        all_features = list(feature_dict.values())

        for feature_vector in zip(*all_features):
            dist = dist_metric(list(feature_vector), x_star)
            all_distances.append((dist, identifier))

    all_distances.sort(key=lambda x: x[0])
    top_k = [identifier for _, identifier in all_distances[:k]]

    counter = Counter(top_k)
    most_common = counter.most_common()

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return all_distances[0][1]

    return most_common[0][0]
