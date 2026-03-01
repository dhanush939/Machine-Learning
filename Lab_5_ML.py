import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

data = pd.read_csv("Data_records.csv")

print("\n========== A1 : LINEAR REGRESSION (Single Attribute) ==========")

X = data[['Duration']]
y = data['Subject age']

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\n========== A2 : METRICS ==========")

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

train_mape = MAPE(y_train, y_train_pred)
test_mape = MAPE(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTRAIN METRICS")
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("MAPE:", train_mape)
print("R2:", train_r2)

print("\nTEST METRICS")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("MAPE:", test_mape)
print("R2:", test_r2)

print("\n========== A3 : MULTIPLE ATTRIBUTE REGRESSION ==========")

X_multi = data[['Duration', 'EEG sample rate', 'Number of EEG channels']]
X_multi = X_multi.fillna(X_multi.mean())

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

model2 = LinearRegression()
model2.fit(X_train_m, y_train_m)

y_train_pred_m = model2.predict(X_train_m)
y_test_pred_m = model2.predict(X_test_m)

train_mse_m = mean_squared_error(y_train_m, y_train_pred_m)
test_mse_m = mean_squared_error(y_test_m, y_test_pred_m)

train_rmse_m = np.sqrt(train_mse_m)
test_rmse_m = np.sqrt(test_mse_m)

train_mape_m = MAPE(y_train_m, y_train_pred_m)
test_mape_m = MAPE(y_test_m, y_test_pred_m)

train_r2_m = r2_score(y_train_m, y_train_pred_m)
test_r2_m = r2_score(y_test_m, y_test_pred_m)

print("\nMULTI FEATURE TRAIN R2:", train_r2_m)
print("MULTI FEATURE TEST R2:", test_r2_m)

print("\n========== A4 : KMEANS (k = 2) ==========")

X_cluster = X_multi.copy()

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans.fit(X_cluster)

labels = kmeans.labels_

print("Cluster Centers:\n", kmeans.cluster_centers_)

print("\n========== A5 : CLUSTER SCORES ==========")

sil = silhouette_score(X_cluster, labels)
ch = calinski_harabasz_score(X_cluster, labels)
db = davies_bouldin_score(X_cluster, labels)

print("Silhouette Score:", sil)
print("Calinski-Harabasz Score:", ch)
print("Davies-Bouldin Index:", db)

print("\n========== A6 : DIFFERENT K VALUES ==========")

k_values = range(2, 11)
sil_scores = []
ch_scores = []
db_scores = []
distortions = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    lbl = km.fit_predict(X_cluster)

    sil_scores.append(silhouette_score(X_cluster, lbl))
    ch_scores.append(calinski_harabasz_score(X_cluster, lbl))
    db_scores.append(davies_bouldin_score(X_cluster, lbl))
    distortions.append(km.inertia_)

plt.figure(figsize=(10,6))
plt.plot(k_values, sil_scores, marker='o', label='Silhouette')
plt.plot(k_values, ch_scores, marker='o', label='CH Score')
plt.plot(k_values, db_scores, marker='o', label='DB Index')
plt.xlabel("k")
plt.ylabel("Score")
plt.title("Scores vs k")
plt.legend()
plt.grid()
plt.show()

print("\n========== A7 : ELBOW METHOD ==========")

plt.figure(figsize=(8,5))
plt.plot(k_values, distortions, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.grid()
plt.show()
