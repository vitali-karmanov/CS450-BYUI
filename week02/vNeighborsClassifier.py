# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:10:28 2019

@author: Vitali Karmanov
"""

import numpy as np

class VNeighborsClassifier:
    
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.X_train = []
        self.y_train = []
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print("Classifier trained using the VNeighborsClassifier model ")
        
        
    def predict(self, X_test):
        
        predictions = []
        
        for X_test_row in X_test:
    
            distance_X_test_row = self.calcDistance(X_test_row)
            
            sorted_res = sorted(distance_X_test_row, key=lambda tup: tup[0])[:self.n_neighbors]
            
            
            list1 = [i[1] for i in sorted_res]
            
            setosa = list1.count(0)
            versicolor = list1.count(1)
            virginica = list1.count(2)
            
            
            prediction = -1
            
            if setosa > versicolor and setosa > virginica:
                prediction = 0
            if versicolor > setosa and versicolor > virginica:
                prediction = 1
            if virginica > versicolor and virginica > setosa:
                prediction = 2
                
            if prediction == -1:
                dist, target = sorted_res[0]
                prediction = target
            
            predictions.append(prediction)
            
        print(predictions)
    
        return predictions
       # print(sorted_res)
        
        #print(np.searchsorted(A,[0,1,2]))
        
        
        
    def calcDistance(self, X_test_row):
    
        distances = []

        for i in range(len(self.X_train)):
            diff = X_test_row - self.X_train[i]
            diff_squared = diff ** 2
            
            dist = sum(diff_squared)
            
            vector = (round(dist, 2), self.y_train[i])
            
            distances.append(vector)
        
        return distances
    
    
    def score(self, X_test, y_test):
        correct = np.mean(self.predict(X_test) == y_test)
        return correct
