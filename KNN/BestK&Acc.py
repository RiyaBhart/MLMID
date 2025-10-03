import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
training = pd.read_csv(r'C:\Users\Me\Desktop\ML LAB MIS\occupancy_train.txt')
testing = pd.read_csv(r'C:\Users\Me\Desktop\ML LAB MIS\occupancy_test.txt')
from sklearn.metrics import accuracy_score

features = ['Humidity','Light','HumidityRatio']
X_train = training[features]
y_train = training['Occupancy']

X_test = testing[features]
y_test = testing['Occupancy']

accuracies=[]

for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracies.append(accuracy)
    print("K : ",i," Accuracy : ",accuracy)
    
best_accuracy = max(accuracies)
print('\nBest accuracy : ',best_accuracy," K = ",accuracies.index(best_accuracy)+1)
