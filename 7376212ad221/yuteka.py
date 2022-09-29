import pandas  as pd
data=pd.read_csv('C:/Users/Student/Documents/7376212ad221/iris (2).csv')
data.head()
X=data.drop(columns=['variety'])
y=data['variety']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
RF=RandomForestClassifier(n_estimators = 100)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
RF=RF.fit(X_train,y_train)
y_pred=RF.predict(X_test)
from sklearn import metrics
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))