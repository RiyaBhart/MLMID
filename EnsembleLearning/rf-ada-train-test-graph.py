
X = df[['restecg', 'chol']]   
y = df['target']              
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)

rf.fit(X_train, y_train)
ada.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

ada_train_acc = accuracy_score(y_train, ada.predict(X_train))
ada_test_acc = accuracy_score(y_test, ada.predict(X_test))
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[
    ('rf', rf),
    ('ada', ada)
], voting='soft')

voting.fit(X_train, y_train)

vote_train_acc = accuracy_score(y_train, voting.predict(X_train))
vote_test_acc = accuracy_score(y_test, voting.predict(X_test))
print("Random Forest → Train:", rf_train_acc, " | Test:", rf_test_acc)
print("AdaBoost      → Train:", ada_train_acc, " | Test:", ada_test_acc)
print("Voting        → Train:", vote_train_acc, " | Test:", vote_test_acc)
import matplotlib.pyplot as plt

models = ['Random Forest','AdaBoost','Voting Ensemble']
train_acc = [rf_train_acc,ada_train_acc,vote_train_acc]
test_acc = [rf_test_acc,ada_test_acc,vote_test_acc]

x = np.arange(len(models))
plt.figure(figsize=(5,4))
plt.bar(x-0.2, train_acc ,width=0.3,label='training acc')
plt.bar(x+0.2,test_acc,width=0.3,label='testing acc')
plt.xticks(x,models)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
     
