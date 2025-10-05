from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


# Use the same dataset as before
X = df[['restecg', 'oldpeak']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define models
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=0)
xgb = XGBClassifier( eval_metric="logloss", random_state=0)

# Voting Classifiers
voting_hard = VotingClassifier(estimators=[('dt', dt), ('knn', knn), ('rf', rf), ('xgb', xgb)], voting='hard')
voting_soft = VotingClassifier(estimators=[('dt', dt), ('knn', knn), ('rf', rf), ('xgb', xgb)], voting='soft')

# Train models
voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

voting_hard_pred = voting_hard.predict(X_test)
voting_soft_pred = voting_soft.predict(X_test)

# Accuracy
voting_hard_acc = accuracy_score(y_test, voting_hard_pred)
voting_soft_acc = accuracy_score(y_test, voting_soft_pred)

print("Voting Classifier (Hard Voting):")
print("Accuracy:", voting_hard_acc)

print("\nVoting Classifier (Soft Voting):")
print("Accuracy:", voting_soft_acc)

weight_options = [1, 2, 3, 4, 5]
best_acc = 0
best_weights = None

# finding best weights
weight_list = [
    [1, 1, 1, 1],
    [1, 2, 2, 1],
    [2, 1, 3, 2],
    [3, 1, 2, 3],
    [2, 2, 1, 3]
]

best_acc = 0
best_weights = None

for w in weight_list:
    voting = VotingClassifier(
        estimators=[('dt', dt), ('knn', knn), ('rf', rf), ('xgb', xgb)],
        voting='soft',
        weights=w
    )
    voting.fit(X_train, y_train)
    acc = accuracy_score(y_test, voting.predict(X_test))
    print("Weights:", w, "→ Accuracy:", acc)
    if acc > best_acc:
        best_acc = acc
        best_weights = w

print("\nBest Weights for Soft Voting:", best_weights)
print("Best Weighted Accuracy:", best_acc)

# bia-variance trade off
final_voting = VotingClassifier(
    estimators=[('dt', dt), ('knn', knn), ('rf', rf), ('xgb', xgb)],
    voting='soft',
    weights=best_weights
)

train_sizes, train_scores, test_scores = learning_curve(
    final_voting, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, label='Training Accuracy')
plt.plot(train_sizes, test_mean, label='Validation Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Bias–Variance Tradeoff (Voting Classifier)')
plt.legend()
plt.grid(True)
plt.show()
