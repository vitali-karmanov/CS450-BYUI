# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
from sklearn import tree
import graphviz 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def startIris():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.10)
    
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_train, y_train)
    
    # Save PDF with Tree
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=iris.feature_names, 
                                    class_names=iris.target_names, filled=True, 
                                    rounded=True, special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("iris")
    
    
    
    y_pred = clf.predict(X_test)
    
    print(y_pred)
    print(y_test)
    
    print ("Accuracy is " + str(accuracy_score(y_test,y_pred)*100))


def startDataset():
    print("What data set would you like to use?")
    option = input("Enter 1 for Iris, or 2 for Automobile MPG, or 3 for StudentPerformance: ")
    
    if option == str(1):
        startIris()

    again = input("\nWould you like to start again? (y/n): ")
    if again == "y":
        startDataset()
        
pd.options.display.max_columns = 20
print("Welcome to Census Data Set example")
startDataset()