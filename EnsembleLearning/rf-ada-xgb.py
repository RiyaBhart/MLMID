import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r'/content/heart.csv')
df.columns
df.info()
df.size
df.describe()
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()

df = pd.get_dummies(df,drop_first=True)

df['target'].value_counts() # balanced
 
# no need to do feature scaling for trees or boosting and bagging

X = df.drop(columns=['target'])
y = df['target']

X_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
X_train_val,X_val,y_train_val,y_val=train_test_split(X_train,y_train,test_size=0.3,random_state=0)
# We use a validation set to evaluate a model’s performance
# during training. It helps in tuning hyperparameters, selecting the best model,
# and preventing overfitting
# by checking how well the model generalizes to unseen data before the final test.

rf = RandomForestClassifier(n_estimators=100,random_state=0)
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=0)

rf.fit(X_train_val, y_train_val)
rf_train_pred = rf.predict(X_train_val)
rf_test_pred = rf.predict(x_test)

rf_train_acc = accuracy_score(y_train_val, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

# AdaBoost
ada.fit(X_train_val, y_train_val)
ada_train_pred = ada.predict(X_train_val)
ada_test_pred = ada.predict(x_test)

ada_train_acc = accuracy_score(y_train_val, ada_train_pred)
ada_test_acc = accuracy_score(y_test, ada_test_pred)

# XGBoost
xgb.fit(X_train_val, y_train_val)
xgb_train_pred = xgb.predict(X_train_val)
xgb_test_pred = xgb.predict(x_test)

xgb_train_acc = accuracy_score(y_train_val, xgb_train_pred)
xgb_test_acc = accuracy_score(y_test, xgb_test_pred)

results = pd.DataFrame({
    'Model': ['Random Forest', 'AdaBoost', 'XGBoost'],
    'Training Accuracy': [rf_train_acc, ada_train_acc, xgb_train_acc],
    'Testing Accuracy': [rf_test_acc, ada_test_acc, xgb_test_acc]
})

print("Model Performance Comparison:")
print(results)

