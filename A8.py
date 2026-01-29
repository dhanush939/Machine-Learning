# data_imputation.py

import pandas as pd
import numpy as np


def load_data(sheet_name):
    return pd.read_excel("Lab Session Data.xlsx", sheet_name=sheet_name)


def has_outliers(values):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return any((values < lower) | (values > upper))


def impute_numeric_column(column):
    values = column.dropna().values

    if has_outliers(values):
        fill_value = np.median(values)
    else:
        fill_value = np.mean(values)

    return column.fillna(fill_value)


def impute_categorical_column(column):
    fill_value = column.mode()[0]
    return column.fillna(fill_value)


def impute_missing_values(data):
    filled_data = data.copy()

    for col in data.columns:
        if data[col].dtype in [int, float]:
            filled_data[col] = impute_numeric_column(data[col])
        else:
            filled_data[col] = impute_categorical_column(data[col])

    return filled_data


def main():
    data = load_data("thyroid0387_UCI")

    print("Missing values before imputation:\n")
    print(data.isnull().sum())

    filled_data = impute_missing_values(data)

    print("\nMissing values after imputation:\n")
    print(filled_data.isnull().sum())


if __name__ == "__main__":
    main()