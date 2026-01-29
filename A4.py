# thyroid_data_exploration.py

import pandas as pd
import numpy as np


def load_data(sheet_name):
    return pd.read_excel("Lab Session Data.xlsx", sheet_name=sheet_name)


def get_column_datatypes(data):
    return data.dtypes


def get_numeric_ranges(data):
    numeric_cols = data.select_dtypes(include=[np.number])
    ranges = {}
    for col in numeric_cols.columns:
        ranges[col] = (numeric_cols[col].min(), numeric_cols[col].max())
    return ranges


def get_missing_value_count(data):
    return data.isnull().sum()


def detect_outliers_iqr(values):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return values[(values < lower) | (values > upper)]


def mean_variance_numeric(data):
    numeric_cols = data.select_dtypes(include=[np.number])
    stats = {}
    for col in numeric_cols.columns:
        stats[col] = {
            "mean": np.mean(numeric_cols[col]),
            "variance": np.var(numeric_cols[col])
        }
    return stats


def main():
    data = load_data("thyroid0387_UCI")

    print("DATA TYPES OF ATTRIBUTES:\n")
    print(get_column_datatypes(data))

    print("\nCATEGORICAL ATTRIBUTE ENCODING SUGGESTION:\n")
    for col in data.columns:
        if data[col].dtype == object:
            unique_vals = data[col].unique()
            if len(unique_vals) <= 5:
                print(f"{col}: Label Encoding (likely ordinal)")
            else:
                print(f"{col}: One-Hot Encoding (nominal)")

    print("\nNUMERIC DATA RANGES:\n")
    ranges = get_numeric_ranges(data)
    for col, value_range in ranges.items():
        print(f"{col}: Min = {value_range[0]}, Max = {value_range[1]}")

    print("\nMISSING VALUES IN EACH COLUMN:\n")
    print(get_missing_value_count(data))

    print("\nOUTLIER INFORMATION (IQR METHOD):\n")
    numeric_data = data.select_dtypes(include=[np.number])
    for col in numeric_data.columns:
        outliers = detect_outliers_iqr(numeric_data[col].dropna().values)
        print(f"{col}: Number of outliers = {len(outliers)}")

    print("\nMEAN AND VARIANCE OF NUMERIC ATTRIBUTES:\n")
    stats = mean_variance_numeric(data)
    for col, values in stats.items():
        print(f"{col}: Mean = {values['mean']}, Variance = {values['variance']}")


if __name__ == "__main__":
    main()