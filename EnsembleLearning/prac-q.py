# -------------------------------
# Q3: Ensemble + Validation
# -------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("heart_disease.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
dt = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
lr = LogisticRegression(max_iter=1000)

# Hard voting
ensemble_hard = VotingClassifier(estimators=[('dt', dt), ('knn', knn), ('lr', lr)], voting='hard')
ensemble_hard.fit(X_train, y_train)
y_pred_hard = ensemble_hard.predict(X_test)
print("Hard Voting Accuracy:", accuracy_score(y_test, y_pred_hard))

# Soft voting
ensemble_soft = VotingClassifier(estimators=[('dt', dt), ('knn', knn), ('lr', lr)], voting='soft')
ensemble_soft.fit(X_train, y_train)
y_pred_soft = ensemble_soft.predict(X_test)
print("Soft Voting Accuracy:", accuracy_score(y_test, y_pred_soft))

# Validation - KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble_soft, X, y, cv=kfold)
print("K-Fold CV Average Accuracy:", cv_scores.mean())

# Validation - Leave-One-Out
loo = LeaveOneOut()
loo_scores = cross_val_score(ensemble_soft, X, y, cv=loo)
print("Leave-One-Out CV Average Accuracy:", loo_scores.mean())
