# Question-A4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_minkowski_distance(vec_a, vec_b, order):
    total = 0
    for i in range(len(vec_a)):
        total += abs(vec_a[i] - vec_b[i]) ** order
    return total ** (1 / order)

file_path = r"C:\Users\dhanush\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
dataset = pd.read_excel(file_path, "Purchase data")

feature_one = dataset.iloc[:, 2].values
feature_two = dataset.iloc[:, 3].values

p_range = range(1, 11)
distance_values = [compute_minkowski_distance(feature_one, feature_two, p) for p in p_range]

plt.plot(p_range, distance_values)
plt.show()
