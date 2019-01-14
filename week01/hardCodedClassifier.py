# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:10:28 2019

@author: Vitali Karmanov
"""

import numpy as np

class HardCodedClassifier:
    
    def fit(self, X_train, y_train):
        print("Classifier trained using the VitaliNB model ")
        
    def predict(self, X_train):
        return ([0 for value in X_train])
        
    def score(self, X_train, y_train):
        correct = np.mean(self.predict(X_train) == y_train)
        return correct
