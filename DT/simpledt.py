import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

df = pd.read_excel(r'/content/loan_approval_dataset.xlsx')
X = pd.get_dummies(df.drop('CLASS',axis=1))
y = df['CLASS']

dt = DecisionTreeClassifier(criterion='entropy',max_depth=1)
dt.fit(X,y)

dot_data = export_graphviz(
    dt, feature_names=list(X.columns),
    filled = True, rounded = True, 
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph
