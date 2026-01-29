# data_normalization.py

import pandas as pd
import numpy as np


def load_data(sheet_name):
    return pd.read_excel("Lab Session Data.xlsx", sheet_name=sheet_name)


def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)


def z_score_normalize(column):
    mean = column.mean()
    std = column.std()
    return (column - mean) / std


def normalize_data(data):
    normalized_data = data.copy()

    numeric_cols = data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # attributes with wide range are normalized
        if data[col].max() - data[col].min() > 1:
            normalized_data[col] = min_max_normalize(data[col])
        else:
            normalized_data[col] = z_score_normalize(data[col])

    return normalized_data


def main():
    data = load_data("thyroid0387_UCI")

    print("Original Numeric Data Summary:\n")
    print(data.describe())

    normalized_data = normalize_data(data)

    print("\nNormalized Numeric Data Summary:\n")
    print(normalized_data.describe())


if __name__ == "__main__":
    main()