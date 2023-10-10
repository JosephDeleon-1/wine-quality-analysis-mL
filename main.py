import numpy as np
import pandas as pd
from collections import Counter
from urllib.request import urlopen
from scipy.stats import zscore

# Load and merge the datasets
red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
red_wine = pd.read_csv(red_url, sep=';')
white_wine = pd.read_csv(white_url, sep=';')
data = pd.concat([red_wine, white_wine], ignore_index=True)

# Split the dataset into features (X) and labels (Y)
X = data.drop('quality', axis=1).values
Y = data['quality'].values

# Data preprocessing
X = zscore(X)  # Normalize the feature values
train_indices = np.random.rand(len(X)) < 0.8
X_train = X[train_indices]
Y_train = Y[train_indices]
X_test = X[~train_indices]
Y_test = Y[~train_indices]


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def get_neighbors(X_train, Y_train, x_test, k):
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    sorted_indices = np.argsort(distances)
    return Y_train[sorted_indices[:k]]


def predict(X_train, Y_train, x_test, k):
    neighbors = get_neighbors(X_train, Y_train, x_test, k)
    return Counter(neighbors).most_common(1)[0][0]


def model_evaluation(X_train, Y_train, X_test, Y_test, k):
    predictions = [predict(X_train, Y_train, x_test, k) for x_test in X_test]
    accuracy = np.sum(Y_test == predictions) / len(Y_test)
    precisions, recalls, f1_scores = [], [], []
    epsilon = 1e-8

    unique_classes = np.unique(Y_test)
    confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)

    for cls in unique_classes:
        tp = np.sum((Y_test == cls) & (predictions == cls))
        fp = np.sum((Y_test != cls) & (predictions == cls))
        fn = np.sum((Y_test == cls) & (predictions != cls))
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1_score = 2 * precision * recall / (precision + recall + epsilon)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        for true_cls in unique_classes:
            confusion_matrix[
                np.where(unique_classes == cls)[0][0], np.where(unique_classes == true_cls)[0][0]] = np.sum(
                (Y_test == true_cls) & (predictions == cls))

    return accuracy, precisions, recalls, f1_scores, confusion_matrix


# Model evaluation
k = 1
accuracy, precisions, recalls, f1_scores, confusion_matrix = model_evaluation(X_train, Y_train, X_test, Y_test, k)
print("Accuracy:", accuracy)
print("Precision:", precisions)
print("Recall:", recalls)
print("F1 Score:", f1_scores)
print("Confusion Matrix:\n", confusion_matrix)

# Grid search for the best k value
best_k = 0
best_accuracy = 0
for k_value in range(1, 21):
    accuracy, _, _, _, _ = model_evaluation(X_train, Y_train, X_test, Y_test, k_value)
    if accuracy > best_accuracy:
        best_k = k_value
        best_accuracy = accuracy

print(f"Best k value (using grid search): {best_k}, with accuracy: {best_accuracy}")


def k_fold_cross_validation(X, Y, k, n_splits=5):
    fold_size = len(X) // n_splits
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    accuracies = []
    for i in range(n_splits):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_test, Y_test = X[test_indices], Y[test_indices]

        accuracy, _, _, _, _ = model_evaluation(X_train, Y_train, X_test, Y_test, k)
        accuracies.append(accuracy)

    return np.mean(accuracies)


# Perform k-fold cross-validation
n_splits = 5
best_k = 0
best_accuracy = 0

for k_value in range(1, 21):
    mean_accuracy = k_fold_cross_validation(X, Y, k_value, n_splits)
    if mean_accuracy > best_accuracy:
        best_k = k_value
        best_accuracy = mean_accuracy

print(f"Best k value (using k-fold cross-validation): {best_k}, with accuracy: {best_accuracy}")


def print_classification_report(unique_classes, precisions, recalls, f1_scores):
    print("\nClassification Report:")
    print(" " * 7, "Precision", " Recall", " F1-score")
    for i, cls in enumerate(unique_classes):
        print(f"Class {cls}: {precisions[i]:.4f} {recalls[i]:.4f} {f1_scores[i]:.4f}")


unique_classes = np.unique(Y_test)
print_classification_report(unique_classes, precisions, recalls, f1_scores)
