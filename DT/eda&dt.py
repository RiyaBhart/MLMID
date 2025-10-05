import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('/content/cancer patient data sets.csv')

# Basic info
print(df.info())
print(df.isnull().sum())

# Check if dataset is balanced
print(df['Level'].value_counts())

# Remove duplicates
df = df.drop_duplicates()

# Fill missing numeric values
df = df.fillna(df.median(numeric_only=True))

# Handle categorical features
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = [col for col in cat_cols if col != 'Level']   # exclude target

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Split features and target
X = df.drop(columns=['Level'])
y = df['Level']

# Train-test split (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Validation split (from train)
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

# Decision Tree model
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(X_train_val, y_train_val)

# Predictions
train_pred = dt.predict(X_train_val)
val_pred = dt.predict(X_val)
test_pred = dt.predict(X_test)

# Accuracy
print('Training Accuracy : ', accuracy_score(y_train_val, train_pred))
print('Validation Accuracy : ', accuracy_score(y_val, val_pred))
print('Testing Accuracy : ', accuracy_score(y_test, test_pred))

# Confusion Matrix & Classification Report
print('\nConfusion Matrix:\n', confusion_matrix(y_test, test_pred))
print('\nClassification Report:\n', classification_report(y_test, test_pred))
