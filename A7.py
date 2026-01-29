# similarity_heatmap.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(sheet_name):
    return pd.read_excel("Lab Session Data.xlsx", sheet_name=sheet_name)


def to_binary(values):
    # converting numeric data to binary (needed for JC and SMC)
    return np.where(values > 0, 1, 0)


def jaccard_coefficient(vec1, vec2):
    a = b = c = 0
    for i in range(len(vec1)):
        if vec1[i] == 1 and vec2[i] == 1:
            a += 1
        elif vec1[i] == 1 and vec2[i] == 0:
            b += 1
        elif vec1[i] == 0 and vec2[i] == 1:
            c += 1
    return a / (a + b + c) if (a + b + c) != 0 else 0


def simple_matching_coefficient(vec1, vec2):
    match = 0
    for i in range(len(vec1)):
        if vec1[i] == vec2[i]:
            match += 1
    return match / len(vec1)


def cosine_similarity(vec1, vec2):
    dot = 0
    mag1 = 0
    mag2 = 0
    for i in range(len(vec1)):
        dot += vec1[i] * vec2[i]
        mag1 += vec1[i] ** 2
        mag2 += vec2[i] ** 2
    return dot / ((mag1 ** 0.5) * (mag2 ** 0.5))


def similarity_matrix(data, method):
    n = len(data)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if method == "JC":
                matrix[i][j] = jaccard_coefficient(data[i], data[j])
            elif method == "SMC":
                matrix[i][j] = simple_matching_coefficient(data[i], data[j])
            elif method == "COS":
                matrix[i][j] = cosine_similarity(data[i], data[j])

    return matrix


def plot_heatmap(matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="coolwarm")
    plt.title(title)
    plt.xlabel("Observation Index")
    plt.ylabel("Observation Index")
    plt.show()


def main():
    data = load_data("thyroid0387_UCI")

    # selecting first 20 observations and all attributes
    first_20 = data.iloc[:20].values

    binary_data = np.array([to_binary(row) for row in first_20])

    jc_matrix = similarity_matrix(binary_data, "JC")
    smc_matrix = similarity_matrix(binary_data, "SMC")
    cos_matrix = similarity_matrix(first_20, "COS")

    print("Jaccard Coefficient Heatmap")
    plot_heatmap(jc_matrix, "Jaccard Coefficient (JC)")

    print("Simple Matching Coefficient Heatmap")
    plot_heatmap(smc_matrix, "Simple Matching Coefficient (SMC)")

    print("Cosine Similarity Heatmap")
    plot_heatmap(cos_matrix, "Cosine Similarity (COS)")


if __name__ == "__main__":
    main()