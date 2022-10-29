import pandas as pd
import numpy as np


# Importing data
diabetes = pd.read_csv('datasets/diabetes_clean.csv')
diabetes.head()


# Importing Library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Import confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

X = diabetes.drop(['diabetes'], axis=1)
# print(X.shape)
# print(X)
y = diabetes['diabetes']
# print(y.shape)
# print(y)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical
num = list(X_train.columns)


# Prepossing Pipeline
pipeline = ColumnTransformer([
    ('num', StandardScaler(), num)
], remainder='drop')


X_train = pipeline.fit_transform(X_train)
y_train = y_train.values

X_test = pipeline.transform(X_test)

# Fitting the Model
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(X_train, y_train)

# # Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# # # Generate the confusion matrix and classification report
print(confusion_matrix(X_test, y_pred))
print(classification_report(X_test, y_pred))


# Import confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from sklearn.datasets import make_classification
tf.random.set_seed(0)

# generate the data
X, y = make_classification(n_classes=2, n_features=4, n_informative=4, n_redundant=0, random_state=42)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# fit the model
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy', 'AUC'])
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), validation_batch_size=64, verbose=0)

# extract the predicted probabilities

p_pred = model.predict(X_test)
p_pred = p_pred.flatten()
# print(p_pred.round(2))
# [1. 0.01 0.91 0.87 0.06 0.95 0.24 0.58 0.78 ...

# extract the predicted class labels
y_pred = np.where(p_pred > 0.5, 1, 0)
# print(y_pred)
# [1 0 1 1 0 1 0 1 1 0 0 0 0 1 1 0 1 0 0 0 0 ...

print(confusion_matrix(y_test, y_pred))
# [[13  1]
#  [ 2  9]]

print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#            0       0.87      0.93      0.90        14
#            1       0.90      0.82      0.86        11
#     accuracy                           0.88        25
#    macro avg       0.88      0.87      0.88        25
# weighted avg       0.88      0.88      0.88        25
