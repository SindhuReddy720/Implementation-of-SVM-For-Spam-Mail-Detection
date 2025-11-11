# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Pelleti Sindhu Sri
RegisterNumber: 212224240113
*/

import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

## Encoding:
<img width="816" height="45" alt="image" src="https://github.com/user-attachments/assets/f4476683-3734-4cfc-9fa1-eed7af89c9fd" />

## Head():
<img width="930" height="303" alt="Screenshot 2025-11-11 153841" src="https://github.com/user-attachments/assets/d4f0269e-7c3f-47d2-9ba1-1d297f859bd3" />

## info():
<img width="451" height="318" alt="Screenshot 2025-11-11 153959" src="https://github.com/user-attachments/assets/365d03ce-6d39-459e-94c3-fe0b6c4d824c" />

## isnull().sum():
<img width="311" height="353" alt="Screenshot 2025-11-11 154030" src="https://github.com/user-attachments/assets/20c69fb0-175b-4a5b-991b-884bca865a69" />

## prediction of y:
<img width="782" height="139" alt="Screenshot 2025-11-11 154200" src="https://github.com/user-attachments/assets/7aab93f6-a1d1-4f15-a9d7-8e9d49e2a80d" />

## Accuracy:
<img width="321" height="90" alt="Screenshot 2025-11-11 154212" src="https://github.com/user-attachments/assets/91b518eb-fc7c-4d46-88d4-28006ae0371e" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
