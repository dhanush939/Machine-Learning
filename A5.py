# similarity_measures.py

import pandas as pd
import numpy as np


def load_data(sheet_name):
    return pd.read_excel("Lab Session Data.xlsx", sheet_name=sheet_name)


def get_binary_vectors(data):
    # selecting first two observations
    first_two = data.iloc[:2]

    # keeping only binary attributes (0/1)
    binary_columns = []
    for col in first_two.columns:
        unique_vals = first_two[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            binary_columns.append(col)

    vec1 = first_two.iloc[0][binary_columns].values
    vec2 = first_two.iloc[1][binary_columns].values

    return vec1, vec2


def calculate_frequencies(vec1, vec2):
    f11 = f10 = f01 = f00 = 0

    for i in range(len(vec1)):
        if vec1[i] == 1 and vec2[i] == 1:
            f11 += 1
        elif vec1[i] == 1 and vec2[i] == 0:
            f10 += 1
        elif vec1[i] == 0 and vec2[i] == 1:
            f01 += 1
        elif vec1[i] == 0 and vec2[i] == 0:
            f00 += 1

    return f11, f10, f01, f00


def jaccard_coefficient(f11, f10, f01):
    return f11 / (f11 + f10 + f01)


def simple_matching_coefficient(f11, f10, f01, f00):
    return (f11 + f00) / (f11 + f10 + f01 + f00)


def main():
    data = load_data("thyroid0387_UCI")

    vec1, vec2 = get_binary_vectors(data)

    f11, f10, f01, f00 = calculate_frequencies(vec1, vec2)

    jc = jaccard_coefficient(f11, f10, f01)
    smc = simple_matching_coefficient(f11, f10, f01, f00)

    print("Binary Vector 1:", vec1)
    print("Binary Vector 2:", vec2)

    print("\nf11:", f11)
    print("f10:", f10)
    print("f01:", f01)
    print("f00:", f00)

    print("\nJaccard Coefficient (JC):", jc)
    print("Simple Matching Coefficient (SMC):", smc)


if __name__ == "__main__":
    main()