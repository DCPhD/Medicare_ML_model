
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

pd.set_option("display.width", 2000)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 2000000)


# K-nearest neighbors model

# read in file
ml_model = pd.read_csv("ML_modelb.csv")
# replace NaN values with 0
ml_model = ml_model.fillna(0)

# Define X and y vectors

# all columns except DC
X = ml_model.iloc[:, 1:].values
# only DC column
y = ml_model.iloc[:, 0].values

# train and test set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# apply feature scaling to normalize the range of each feature
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# apply classifier to train data
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# apply classifier to test data
y_pred = classifier.predict(X_test)

# evaluating algorithm
#print(accuracy_score(y_test, y_pred, normalize = True))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''
confusion matrix

[[192813    971]
 [  1498   5305]]
 
classification report

              precision    recall  f1-score   support

           0       0.99      0.99      0.99    193784
           1       0.85      0.78      0.81      6803

    accuracy                           0.99    200587
   macro avg       0.92      0.89      0.90    200587
weighted avg       0.99      0.99      0.99    200587
'''

## Gaussian NB model

# read in file
ml_modelNB = pd.read_csv("ML_modelb.csv")
# replace NaN values with 0
ml_modelNB = ml_modelNB.fillna(0)

# Define X and y vectors

# all columns except DC
X = ml_modelNB.iloc[:, 1:].values
# only DC column
y = ml_modelNB.iloc[:, 0].values

# train and test set splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# apply feature scaling to normalize the range of each feature
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# apply classifier to train data
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# apply classifier to test data
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


'''
# confusion matrix
[[183409  10318]
 [   248   6612]]

# classification table

              precision    recall  f1-score   support

           0       1.00      0.95      0.97    193727
           1       0.39      0.96      0.56      6860

    accuracy                           0.95    200587
   macro avg       0.69      0.96      0.76    200587
weighted avg       0.98      0.95      0.96    200587
'''