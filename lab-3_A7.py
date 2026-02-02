# Question-A7
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

file_path = r"C:\Users\dhanush\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
dataset = pd.read_excel(file_path, "Purchase data")

features = dataset.iloc[:, 1:4].values
targets = np.where(dataset.iloc[:, 4].values < 250, 0, 1)

features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.3, random_state=42
)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(features_train, targets_train)

print("Training completed")
