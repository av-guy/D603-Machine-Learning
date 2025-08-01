from random import seed

from ..linear_regression import (
    evaluate_mlr,
    predict_multiple_linear_regression,
    linear_regression_sgd,
    normalize_dataset,
    dataset_minmax
)

from ..files import load_csv

N_FOLDS = 5
L_RATE = 0.01
N_EPOCH = 50


def main():
    """ Run the demo """
    seed(1)

    wine = load_csv("data", "wine.csv", delimiter=";")
    wine = {k: list(map(float, v)) for k, v in wine.items()}

    min_max_vals = dataset_minmax(wine)
    normalized_wine = normalize_dataset(wine, min_max_vals)

    x = [tuple(features) for features in zip(
        *[v for k, v in normalized_wine.items() if k != "quality"])]

    y = normalized_wine["quality"]

    scores = evaluate_mlr(
        x,
        y,
        train_func=linear_regression_sgd,
        predict_func=predict_multiple_linear_regression,
        n_folds=N_FOLDS,
        n_epoch=N_EPOCH,
        l_rate=L_RATE
    )

    print(f"Cross-validated RMSE scores: {scores}")
    print(f"Mean RMSE: {sum(scores) / len(scores):.3f}")
