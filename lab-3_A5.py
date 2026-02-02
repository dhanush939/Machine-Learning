# Question-A5
import pandas as pd
from scipy.spatial.distance import minkowski

def calculate_minkowski_distance(vec_one, vec_two, order):
    total = 0
    for i in range(len(vec_one)):
        total += abs(vec_one[i] - vec_two[i]) ** order
    return total ** (1 / order)

file_path = r"C:\Users\dhanush\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
dataset = pd.read_excel(file_path, "Purchase data")

feature_x = dataset.iloc[:, 2].values
feature_y = dataset.iloc[:, 3].values

print(calculate_minkowski_distance(feature_x, feature_y, 2))
print(minkowski(feature_x, feature_y, 2))
