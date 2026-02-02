# Question-A3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_location = r"C:\Users\dhanush\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
dataset = pd.read_excel(file_location, "Purchase data")

selected_column = dataset.iloc[:, 1].values

histogram_data = np.histogram(selected_column)
average_value = np.mean(selected_column)
variance_value = np.var(selected_column)

plt.hist(selected_column)
plt.show()

print(histogram_data)
print(average_value)
print(variance_value)
