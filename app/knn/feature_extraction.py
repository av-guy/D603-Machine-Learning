"""
feature_extraction.py

Feature extraction utility for CSV datasets.

All extracted values are converted to float, and rows with invalid or missing
numerical data are skipped.

Functions
---------
extract_features(file_name, feature_names, identifier_name)
    Extracts specified numeric columns from a CSV file and groups them by a
    categorical identifier column.
"""

from typing import List, Tuple, Dict, Union
from csv import DictReader


def extract_features(
    file_name: str,
    feature_names: Union[List[str], Tuple[str, ...]],
    identifier_name: str
) -> Dict[str, Dict[str, List[float]]]:
    """
    Extracts numerical feature values from a CSV file, grouped by a categorical identifier.

    Parameters
    ----------
    file_name : str
        Path to the CSV file.
    feature_names : Union[List[str], Tuple[str, ...]]
        Names of the numerical columns to extract.
    identifier_name : str
        The column used to group rows (e.g., "Species", "Island", etc.).

    Returns
    -------
    Dict[str, Dict[str, List[float]]]
        A dictionary where keys are unique identifier values from the file (e.g., "Adelie"),
        and values are dictionaries mapping each feature name to a list of float values.
    """
    feature_dict: Dict[str, Dict[str, List[float]]] = {}

    with open(file_name, encoding="utf-8") as f:
        reader = DictReader(f)

        for row in reader:
            identifier = row[identifier_name]

            entry = feature_dict.setdefault(
                identifier, {name: [] for name in feature_names})

            try:
                for name in feature_names:
                    entry[name].append(float(row[name]))
            except ValueError:
                # skip rows with missing or invalid numeric data
                continue

    return feature_dict
