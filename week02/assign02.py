# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 10:09:06 2019

@author: Vitali Karmanov
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


from vNeighborsClassifier import VNeighborsClassifier


def startIris():

    # Load Iris datasets from sklearn
    iris = datasets.load_iris()
    
    # Shuffle datasets, and split into train and test (6 items to test).
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.10)
    
    #Load own classifier model 
    classifierVitali = VNeighborsClassifier(n_neighbors=3)
    
    # Train the classifier using your own model 
    classifierVitali.fit(X_train, y_train)
    
    predictionV = classifierVitali.predict(X_test)
    
    # Test the classifier with the test dataset.
    score = classifierVitali.score(X_test, y_test)
   
    print("Score: ")
    print(score)
    
    print("Actual: ")
    print(y_test)
    
    
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    
    print("Prediction Vitali: ")
    print(predictionV)
    
    print("Oficial: ")
    print(predictions)
    
    
    # Print the % of accuracy 
    #print("Accuracy: " + str(classifierVitali.score(X_test, y_test)*100) + "%")


print("Welcome to Iris Data Set example")
startIris()