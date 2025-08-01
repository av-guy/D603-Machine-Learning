from os import path

from csv import DictReader
from typing import Dict, Any, Callable, Optional, List


def load_csv(
    file_path: str,
    file_name: str,
    converters: Optional[Dict[str, Callable[[str], Any]]] = None,
    delimiter: Optional[str] = ","
) -> Dict[str, List[Any]]:
    """
    Load a CSV file and return a dictionary where each column maps to a list of its values.

    Parameters
    ----------
    file_path : str
        Directory path to the CSV file.
    file_name : str
        Name of the CSV file (e.g., "data.csv").
    converters : Optional[Dict[str, Callable[[str], Any]]]
        Dictionary mapping column names to conversion functions.
    delimiter: Optional[str] = ","
        An optional delimiter if different from ','.

    Returns
    -------
    Dict[str, List[Any]]
        Dictionary where each key is a column name and each value is a list of parsed values.
    """
    full_path = path.join(file_path, file_name)
    result: Dict[str, List[Any]] = {}

    with open(full_path, newline='', encoding="utf-8") as f:
        csv_reader = DictReader(f, delimiter=delimiter)

        for row in csv_reader:
            for key, value in row.items():
                key = key.strip()
                if converters and key in converters:
                    value = converters[key](value)
                result.setdefault(key, []).append(value)

    return result
