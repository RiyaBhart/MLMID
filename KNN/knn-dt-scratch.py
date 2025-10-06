# -------------------------------
# Q2: KNN and Decision Tree (from scratch)
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Load dataset
data = pd.read_csv("iris.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- KNN from Scratch ----------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for x in X_test:
        distances = []
        for i, x_train in enumerate(X_train):
            distance = euclidean_distance(x, x_train)
            distances.append((distance, y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = [label for (_, label) in distances[:k]]
        prediction = Counter(k_nearest).most_common(1)[0][0]
        predictions.append(prediction)
    return np.array(predictions)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

y_pred_knn = knn_predict(X_train, y_train, X_test, k=3)
print("KNN Accuracy:", accuracy(y_test, y_pred_knn))

# ---------- Decision Tree from Scratch ----------
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini(y):
    classes = np.unique(y)
    impurity = 1
    for c in classes:
        p = np.sum(y == c) / len(y)
        impurity -= p**2
    return impurity

def split_dataset(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def best_split(X, y):
    best_feature, best_threshold, best_gain = None, None, 0
    current_impurity = gini(y)
    n_features = X.shape[1]
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split_dataset(X, y, feature, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            p = len(y_left) / len(y)
            gain = current_impurity - (p * gini(y_left) + (1 - p) * gini(y_right))
            if gain > best_gain:
                best_feature, best_threshold, best_gain = feature, threshold, gain
    return best_feature, best_threshold

def build_tree(X, y, depth=0, max_depth=5):
    if len(np.unique(y)) == 1 or depth >= max_depth:
        return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])
    feature, threshold = best_split(X, y)
    if feature is None:
        return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])
    X_left, X_right, y_left, y_right = split_dataset(X, y, feature, threshold)
    left = build_tree(X_left, y_left, depth + 1, max_depth)
    right = build_tree(X_right, y_right, depth + 1, max_depth)
    return DecisionTreeNode(feature, threshold, left, right)

def predict_tree(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

tree = build_tree(X_train, y_train)
y_pred_tree = [predict_tree(tree, x) for x in X_test]
print("Decision Tree Accuracy:", accuracy(y_test, y_pred_tree))
