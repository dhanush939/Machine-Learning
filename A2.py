# purchase_classifier.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def load_data(file_path, sheet_name):
    """Load purchase data from Excel."""
    return pd.read_excel(file_path, sheet_name=sheet_name)


def prepare_features(data):
    """Extract feature matrix from purchase data."""
    return data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values


def create_labels(data, threshold=200):
    """
    Create class labels:
    1 -> RICH (payment > threshold)
    0 -> POOR (payment <= threshold)
    """
    return np.where(data["Payment (Rs)"] > threshold, 1, 0)


def train_classifier(X, labels):
    """Train a logistic regression classifier."""
    model = LogisticRegression()
    model.fit(X, labels)
    return model


def predict_classes(model, X):
    """Predict customer classes."""
    return model.predict(X)


def main():
    file_path = "Lab Session Data.xlsx"
    sheet_name = "Purchase Data"

    data = load_data(file_path, sheet_name)

    X = prepare_features(data)
    y = create_labels(data)

    classifier = train_classifier(X, y)
    predictions = predict_classes(classifier, X)

    class_names = {1: "RICH", 0: "POOR"}

    print("Customer Classification Results:\n")
    for index, predicted_class in enumerate(predictions):
        print(f"Customer {index + 1}: {class_names[predicted_class]}")


if __name__ == "__main__":
    main()