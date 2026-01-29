# cosine_similarity.py

import pandas as pd
import numpy as np


def load_data(sheet_name):
    return pd.read_excel("Lab Session Data.xlsx", sheet_name=sheet_name)


def get_vectors(data, index1, index2):
    """
    Extract complete feature vectors for two observations.
    Assumes all columns are numeric or already encoded.
    """
    vector_a = data.iloc[index1].values
    vector_b = data.iloc[index2].values
    return vector_a, vector_b


def dot_product(vec1, vec2):
    total = 0
    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]
    return total


def vector_magnitude(vector):
    total = 0
    for value in vector:
        total += value ** 2
    return total ** 0.5


def cosine_similarity(vec1, vec2):
    numerator = dot_product(vec1, vec2)
    denominator = vector_magnitude(vec1) * vector_magnitude(vec2)
    return numerator / denominator


def main():
    data = load_data("thyroid0387_UCI")

    # selecting two observations (example: first and second rows)
    vector_1, vector_2 = get_vectors(data, 0, 1)

    similarity = cosine_similarity(vector_1, vector_2)

    print("Vector 1:", vector_1)
    print("Vector 2:", vector_2)
    print("\nCosine Similarity between the two observations:", similarity)


if __name__ == "__main__":
    main()