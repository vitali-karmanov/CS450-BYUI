# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 10:09:06 2019

@author: Vitali Karmanov
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from hardCodedClassifier import HardCodedClassifier

def startIris():

    # Load Iris datasets from sklearn
    iris = datasets.load_iris()
    
    # Shuffle datasets, and split into train and test (70% train and 30% test).
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
    
    print("What Model would you like to use?")
    option = input("Enter 1 for GaussianNB, or 2 for VitaliNB: ")
    
    if option == str(1):
    
        # Train the classifier using GaussianNB model 
        print("Classifier trained using the GaussianNB model ")
        classifierGaussian = GaussianNB()
        classifierGaussian.fit(X_train, y_train)
    
        # Test the classifier with the test dataset.
        targetsPredictedGaussian = classifierGaussian.predict(X_test)
        print("Prediction: ")
        print(targetsPredictedGaussian)
    
        # Print the % of accuracy 
        print("Accuracy: " + str(classifierGaussian.score(X_test, y_test)*100) + "%")
    
    if option == str(2):
        #Load own classifier model 
        classifierVitali = HardCodedClassifier()
    
        # Train the classifier using your own model 
        classifierVitali.fit(X_train, y_train)
    
        # Test the classifier with the test dataset.
        targetsPredictedVitali = classifierVitali.predict(y_test)
        print("Prediction: ")
        print(targetsPredictedVitali)
    
        # Print the % of accuracy 
        print("Accuracy: " + str(classifierVitali.score(X_test, y_test)*100) + "%")

    
    again = input("\nWould you like to start again? (y/n): ")
    if again == "y":
        startIris()
        
print("Welcome to Iris Data Set example")
startIris()