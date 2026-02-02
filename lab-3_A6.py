# Question-A6
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

file_path = r"C:\Users\dhanush\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
dataset = pd.read_excel(file_path, "Purchase data")

features = dataset.iloc[:, 1:4].values
labels = np.where(dataset.iloc[:, 4].values < 250, 0, 1)

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

print(features_train)
print(features_test)
print(labels_train)
print(labels_test)
