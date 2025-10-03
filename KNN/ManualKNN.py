import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
iris = load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
import numpy as np
from collections import Counter

def manhattan_distance(x,y):
    return np.sum(np.abs(x-y))

class KNNClassifier:
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        
    def predict_one(self,x):
        distances = [manhattan_distance(x,X_train) for X_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]
        
        k_labels = [self.y_train[i] for i in k_indices]
        
        return Counter(k_labels).most_common(1)[0][0]
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]
    
    knn = KNNClassifier(k=5)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print('Accuracy : ',accuracy_score(y_test,y_pred))
print('\nConfusion matrix : \n',confusion_matrix(y_test,y_pred))
