# purchase_data.py

import pandas as pd
import numpy as np


def load_data(file_path, sheet_name):
    """Read purchase data from Excel."""
    return pd.read_excel(file_path, sheet_name=sheet_name)


def split_features_and_output(data):
    """Split data into feature matrix X and output vector y."""
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = data["Payment (Rs)"].values
    return X, y


def get_rank(matrix):
    """Return rank of a matrix."""
    return np.linalg.matrix_rank(matrix)


def get_costs(X, y):
    """Compute product costs using pseudo-inverse."""
    X_pinv = np.linalg.pinv(X)
    return X_pinv @ y


def main():
    file_path = "Lab Session Data.xlsx"
    sheet_name = "Purchase Data"

    data = load_data(file_path, sheet_name)
    X, y = split_features_and_output(data)

    dimensions = X.shape[1]      # number of features
    total_vectors = X.shape[0]   # number of observations
    rank = get_rank(X)
    costs = get_costs(X, y)

    print("Dimensionality of vector space:", dimensions)
    print("Number of vectors:", total_vectors)
    print("Rank of feature matrix:", rank)

    print("\nEstimated cost of each product:")
    print("Candy (Rs):", costs[0])
    print("Mangoes per Kg (Rs):", costs[1])
    print("Milk Packet (Rs):", costs[2])


if __name__ == "__main__":
    main()