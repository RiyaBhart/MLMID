import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r'C:\Users\Me\Desktop\ML LAB MIS\cancer patient data sets (1).csv')
df.columns

df.isnull().sum()

df.duplicated().sum()

df.describe()

from sklearn.preprocessing import LabelEncoder, StandardScaler

df_encoded = df.copy()

df_encoded = df_encoded.drop(columns=['index', 'Patient Id'])

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

corr_matrix = df_encoded.corr(method='pearson')

print("\nCorrelation with target (Level):")
print(corr_matrix['Level'].sort_values(ascending=False))

selected_features = corr_matrix['Level'].drop('Level').abs()
selected_features = selected_features[selected_features > 0.2].index.tolist()
print("\nSelected features:", selected_features)

X = df_encoded[selected_features]
y = df_encoded['Level']

from sklearn.model_selection import train_test_split

scaler = StandardScaler()

random_state = 0
X_train_val,X_test,y_train_val,y_test = train_test_split(X,y,test_size=0.2,random_state=random_state, stratify=y)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.30, random_state=random_state, stratify=y_train_val
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

from sklearn.metrics import confusion_matrix

for metric in ["euclidean", "manhattan", "chebyshev"]:
    model = KNeighborsClassifier(n_neighbors=5, metric=metric)
    model.fit(X_train_s, y_train)

    print(f"\nMetric: {metric}")
    print("Train acc:", accuracy_score(y_train, model.predict(X_train_s)))
    print("Val acc:", accuracy_score(y_val, model.predict(X_val_s)))
    print("Test acc:", accuracy_score(y_test, model.predict(X_test_s)))
    
cm = confusion_matrix(y_test, model.predict(X_test_s))
print("\nConfusion Matrix:\n", cm)


