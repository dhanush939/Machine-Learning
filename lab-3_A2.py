# Question-A2
import pandas as pd
import numpy as np

file_path = r"C:\Users\dhanush\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
dataset = pd.read_excel(file_path, "Purchase data")

features = dataset.iloc[:, 1:4].values
amount_paid = dataset.iloc[:, 4].values

group_low = features[amount_paid < 250]
group_high = features[amount_paid >= 250]

center_low = group_low.mean(axis=0)
center_high = group_high.mean(axis=0)

spread_low = group_low.std(axis=0)
spread_high = group_high.std(axis=0)

centroid_distance = np.linalg.norm(center_low - center_high)

print(center_low)
print(spread_low)
print(center_high)
print(spread_high)
print(centroid_distance)
