import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier      
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier

train_data=pd.read_csv('train.csv')
for i in range(len(train_data.columns)):
  train_data[train_data.columns[i]].fillna(value=train_data[train_data.columns[i]].mean(),inplace=True)

y=train_data['PerStatus'].astype(int)
X=train_data.drop(['PerStatus','PerNo'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X.iloc[:,:],y,test_size=0.3,random_state=0)
ms=MinMaxScaler()
ss=StandardScaler()                          
X_train=ms.fit_transform(X_train)
X_test=ms.transform(X_test)

model_xgb=XGBClassifier()
model_xgb.fit(X_train,y_train)

model_rfc=RandomForestClassifier()
model_rfc.fit(X_train,y_train)

model_DT=DecisionTreeClassifier()
model_DT.fit(X_train,y_train)

model=SGDClassifier()
model.fit(X_train,y_train)

print(f'訓練分數:{model.score(X_test,y_test)}')


test_data=pd.read_csv('test.csv')
for i in range(len(test_data.columns)):
  test_data[test_data.columns[i]].fillna(value=test_data[test_data.columns[i]].mean(),inplace=True)
test_X=test_data.drop(['PerStatus','PerNo'],axis=1)

y_pred=model.predict(test_X)
print(y_pred.sum())
No=pd.read_csv('test.csv')
ans={'PerNo':No['PerNo'],'PerStatus':y_pred}
data_F=pd.DataFrame(ans)
data_F.to_csv('SGDClassifier.csv', index=False)