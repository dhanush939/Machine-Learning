import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# ---------------- A1 : CONFUSION MATRIX + METRICS ---------------- #

def make_conf_matrix(actual_vals, predicted_vals):
    tp = np.sum((actual_vals == 1) & (predicted_vals == 1))
    tn = np.sum((actual_vals == 0) & (predicted_vals == 0))
    fp = np.sum((actual_vals == 0) & (predicted_vals == 1))
    fn = np.sum((actual_vals == 1) & (predicted_vals == 0))

    return np.array([[tn, fp], [fn, tp]])


def get_classification_stats(matrix):
    tn, fp = matrix[0]
    fn, tp = matrix[1]

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


def analyze_learning(train_stats, test_stats, slight=0.05, over=0.15):
    tr = train_stats['accuracy']
    te = test_stats['accuracy']
    diff = tr - te

    if tr < 0.6 and te < 0.6:
        status = "UNDERFIT"
        msg = "Model too simple."
    elif diff < slight:
        status = "GOOD FIT"
        msg = "Training and testing accuracies are similar."
    elif diff < over:
        status = "MILD OVERFIT"
        msg = "Training accuracy slightly higher than testing."
    else:
        status = "OVERFITTING"
        msg = "Model memorizing training data."

    return {"fit_type": status, "train_accuracy": tr, "test_accuracy": te, "difference": diff, "explanation": msg}


def prepare_classification_dataset(file_name):
    dataset = pd.read_csv(file_name)

    input_cols = ['Subject age', 'Duration']
    target_col = 'Experience'

    clean_data = dataset[input_cols + [target_col]].dropna()

    encoder = LabelEncoder()
    clean_data['encoded_target'] = encoder.fit_transform(clean_data[target_col])

    features = clean_data[input_cols].values
    labels = clean_data['encoded_target'].values

    train_X, test_X, train_y, test_y = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )

    return train_X, test_X, train_y, test_y


# ---------------- A2 : REGRESSION METRICS ---------------- #

def mse_value(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse_value(y_true, y_pred):
    return np.sqrt(mse_value(y_true, y_pred))


def mape_value(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_value(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def load_regression_data(file_name):
    data = pd.read_excel(file_name, sheet_name="Purchase data")

    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = data["Payment (Rs)"].values

    pinv = np.linalg.pinv(X)
    costs = pinv @ y
    y_pred = X @ costs

    return y, y_pred


# ---------------- A3 : SYNTHETIC DATA ---------------- #

def create_synthetic_points(count=20, value_range=(1, 10), seed=42):
    np.random.seed(seed)
    pts = np.random.uniform(value_range[0], value_range[1], size=(count, 2))
    return pts


def label_points(points, mode='linear'):
    if mode == 'linear':
        labels = (points[:, 0] > 5.5).astype(int)
    elif mode == 'circular':
        center = np.array([5.5, 5.5])
        dist = np.sqrt(np.sum((points - center) ** 2, axis=1))
        labels = (dist > 3.0).astype(int)
    elif mode == 'diagonal':
        labels = (points[:, 0] + points[:, 1] > 11).astype(int)
    else:
        raise ValueError("Invalid mode")
    return labels


def plot_points(X, y, title='Training Data', save=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'red']

    for c in [0, 1]:
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=100, edgecolors='black')

    ax.set_title(title)
    ax.grid(True)

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    return fig


# ---------------- A4 : kNN ---------------- #

def create_prediction_grid(xlim=(0, 10), ylim=(0, 10), step=0.1):
    xvals = np.arange(xlim[0], xlim[1] + step, step)
    yvals = np.arange(ylim[0], ylim[1] + step, step)
    gx, gy = np.meshgrid(xvals, yvals)
    grid = np.c_[gx.ravel(), gy.ravel()]
    return grid, gx, gy


def build_knn_model(train_X, train_y, neighbors=3):
    knn_model = KNeighborsClassifier(n_neighbors=neighbors)
    knn_model.fit(train_X, train_y)
    return knn_model


def draw_boundary(model, X, y, gx, gy, k, save=None):
    grid = np.c_[gx.ravel(), gy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(gx, gy, Z, alpha=0.3, cmap='coolwarm')

    for c in [0, 1]:
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], s=90, edgecolors='black')

    ax.set_title(f"kNN Decision Boundary (k={k})")

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    return fig


# ---------------- MAIN ---------------- #

def main():

    print("kNN MACHINE LEARNING LAB")

    train_X, test_X, train_y, test_y = prepare_classification_dataset('Data records.csv')

    model = build_knn_model(train_X, train_y, neighbors=5)

    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)

    cm_train = make_conf_matrix(train_y, train_pred)
    cm_test = make_conf_matrix(test_y, test_pred)

    train_stats = get_classification_stats(cm_train)
    test_stats = get_classification_stats(cm_test)

    print("Train Accuracy:", train_stats['accuracy'])
    print("Test Accuracy:", test_stats['accuracy'])

    # Synthetic data
    syn_X = create_synthetic_points()
    syn_y = label_points(syn_X, mode='diagonal')

    plot_points(syn_X, syn_y, "Synthetic Data", "a3_training_data.png")

    grid, gx, gy = create_prediction_grid()

    knn3 = build_knn_model(syn_X, syn_y, neighbors=3)
    draw_boundary(knn3, syn_X, syn_y, gx, gy, 3, "a4_decision_boundary_k3.png")

    plt.close('all')


if __name__ == "__main__":
    main()

