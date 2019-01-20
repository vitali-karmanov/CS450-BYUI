# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 10:09:06 2019

@author: Vitali Karmanov
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


from vNeighborsClassifier import VNeighborsClassifier


def startIris():
    

    # Load Iris datasets from sklearn  
    iris = datasets.load_iris()
    
    print("\nWhat percentage would you like to use for your test data?")
    option = input("For example enter 10 for 10%, 30 for 30% and so on. Percentage: ")

    # Shuffle datasets, and split into train and test.
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=int(option)/100)
    
    #Load own classifier model 
    classifierVitali = VNeighborsClassifier(n_neighbors=3)
    
    # Train the classifier using your own model 
    classifierVitali.fit(X_train, y_train)
    
    predictionV = classifierVitali.predict(X_test)

    
    print("Actual Target Test: ")
    print(y_test)

    
    print("Prediction VitaliNeighborsClassifier: ")
    print(predictionV)
    
    # Test the classifier with the test dataset. (New prediction is called when the Score function is called)
    print("Score: " + str(classifierVitali.score(X_test, y_test)*100) + "%")
    
    
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    
    print("Prediction KNeighborsClassifier: ")
    print(predictions)
    
    again = input("\nWould you like to start again? (y/n): ")
    if again == "y":
        startIris()



print("Welcome to Iris Data Set example")
startIris()
