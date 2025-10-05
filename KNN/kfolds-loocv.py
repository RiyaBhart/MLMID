kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(voting, X, y, cv=kfold)

print("K-Fold Cross Validation Scores:", kfold_scores)
print("Average K-Fold Accuracy:", np.mean(kfold_scores))

loocv = LeaveOneOut()
loocv_scores = cross_val_score(voting, X, y, cv=loocv)

print("LOOCV Average Accuracy:", np.mean(loocv_scores))
