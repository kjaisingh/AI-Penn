# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

for i in range(0, len(y)):
	if(y[i] == "Iris-setosa"):
		y[i] = 1
	if(y[i] == "Iris-versicolor"):
		y[i] = 2
	if(y[i] == "Iris-virginica"):
		y[i] = 3

X = X.astype(float)
y = y.astype(int)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Printing the final accuracy
from sklearn.metrics import accuracy_score
print("Final accuracy is: ", accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)