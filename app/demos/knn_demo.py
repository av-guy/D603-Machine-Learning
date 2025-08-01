from ..knn import extract_features, k_nearest_neigbhors
from ..stats import std, mean

X_STAR = [0.2, -1]
FEATURES = ["Culmen Length (mm)", "Culmen Depth (mm)"]


def main():
    """ Run the demo """
    penguin_data = extract_features(
        file_name="data/penguins_lter.csv",
        feature_names=FEATURES,
        identifier_name="Species"
    )

    for _, values in penguin_data.items():
        for feature_name, feature_set in values.items():
            x_bar = mean(feature_set)
            sigma = std(feature_set)

            scaled = list(map(lambda x, x_bar=x_bar, sigma=sigma: (
                x - x_bar) / sigma, feature_set))

            values[feature_name] = scaled

    prediction = k_nearest_neigbhors(
        x_star=X_STAR,
        identifiers_dict=penguin_data,
        k=9,
        metric="manhattan"
    )

    print(f"Predicted species: {prediction}")
