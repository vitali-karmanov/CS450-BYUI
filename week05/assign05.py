# -*- coding: utf-8 -*-
"""
Vitali Karmanov

"""

import graphviz 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


def startIris():
    iris = load_iris()

    print("\nWhat percentage would you like to use for your test data?")
    test_option = input("For example enter 10 for 10%, 30 for 30% and so on. Percentage: ")

    # Shuffle datasets, and split into train and test.
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=int(test_option)/100)
    
    print("\nWhat is the max-depth of the tree?")
    max_depth = input("max-depth: ")

    clf = tree.DecisionTreeClassifier(max_depth=int(max_depth))
    clf.fit(X_train, y_train)
    
    
    # Save PDF with Tree
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=iris.feature_names, 
                                    class_names=iris.target_names, filled=True, 
                                    rounded=True, special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("iris")
    print("\nSaved PDF with tree, see the folder\n")
    
    # Predict
    y_pred = clf.predict(X_test)

    #Results
    print("Predicted: ")
    print(y_pred)
    print("Actual: ")
    print(y_test)
    
    # Compute and print the accuracy
    print ("\nAccuracy is " + str(accuracy_score(y_test,y_pred)*100))
    
    
def startAutomobileMPG():
    
    names = ["mpg","cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year","origin","car_name"]
  
    data = pd.read_csv("auto-mpg.data.csv", header=None,
                   names=names, na_values=["?"], delim_whitespace=True)
    
    data.horsepower = data.horsepower.fillna(data.horsepower.median())
    
    data = data.drop(columns=["car_name"])
    
    X = data.drop(columns=["mpg"]).values
    y = data["mpg"].values.flatten()
    
    print("\nWhat percentage would you like to use for your test data?")
    test_option = input("For example enter 10 for 10%, 30 for 30% and so on. Percentage: ")

    # Shuffle datasets, and split into train and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(test_option)/100)
    
    print("\nWhat is the max-depth of the tree?")
    max_depth = input("max-depth: ")

    regr = DecisionTreeRegressor(max_depth=int(max_depth))
    regr.fit(X_train, y_train)
    
    feature_names = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year","origin"]
    
    target_names = ["mpg"]
    
    # Save PDF with Tree
    dot_data = tree.export_graphviz(regr, out_file=None, 
                                    feature_names=feature_names,
                                    class_names=target_names, filled=True, 
                                    rounded=True, special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("AutomobileMPG-tree")
    print("\nSaved PDF with tree, see the folder\n")
    
    
    #Predict
    y_pred = regr.predict(X_test)


    # Plot image
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Ground Truth vs Predicted")
    plt.show()
    fig.savefig('AutomobileMPG-plot.jpg')
    
    #Results
    print("Predicted: ")
    print(y_pred)
    print("Actual: ")
    print(y_test)
    

    # Compute and print the accuracy
    accuracy = r2_score(y_test, y_pred)
    
    print("\nAutomobile MPG - Accuracy: {}".format(str(accuracy*100) + "%"))
    


def startStudentPerformance():
      
    data = pd.read_csv("student-mat.csv", sep=";")
    
    data["isMale"] = data.sex.map({"M": 1, "F": 0})
    data["isGabrielPereira"] = data.school.map({"GP": 1, "MS": 0})
    data["isUrban"] = data.address.map({"U": 1, "R": 0})
    data["isLessOrEqual"] = data.famsize.map({"LE3": 1, "GT3": 0})
    data["isLivingTogether"] = data.Pstatus.map({"T": 1, "A": 0})

    data.Mjob = data.Mjob.astype('category')
    data["Mjob_cat"] = data.Mjob.cat.codes
    
    data.Fjob = data.Fjob.astype('category')
    data["Fjob_cat"] = data.Fjob.cat.codes

    data.reason = data.Fjob.astype('category')
    data["reason_cat"] = data.reason.cat.codes

    data.guardian = data.Fjob.astype('category')
    data["guardian_cat"] = data.guardian.cat.codes

    data["isExtraEducationalSupport"] = data.schoolsup.map({"yes": 1, "no": 0})
    data["isFamilyEducationalSupport"] = data.famsup.map({"yes": 1, "no": 0})
    data["isExtraPaidClasses"] = data.paid.map({"yes": 1, "no": 0})
    data["isExtraCurricularActivities"] = data.activities.map({"yes": 1, "no": 0})
    data["isAttendedNurserySchool"] = data.nursery.map({"yes": 1, "no": 0})
    data["wantsHigheEducation"] = data.higher.map({"yes": 1, "no": 0})
    data["internetAccessAtHome"] = data.internet.map({"yes": 1, "no": 0})
    data["withRomanticRelationship"] = data.romantic.map({"yes": 1, "no": 0})
    
    data = data.drop(columns=["school", "sex", "address", "famsize",
                                          "Pstatus", "Mjob", "Fjob", "reason",
                                          "guardian", "schoolsup", "famsup",
                                          "paid", "activities", "nursery",
                                          "higher", "internet", "romantic"])
    
    X = data.drop(columns=["G3"]).values
    y = data["G3"].values.flatten()
    
    
    print("\nWhat percentage would you like to use for your test data?")
    test_option = input("For example enter 10 for 10%, 30 for 30% and so on. Percentage: ")

    # Shuffle datasets, and split into train and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(test_option)/100)
    
    print("\nWhat is the max-depth of the tree?")
    max_depth = input("max-depth: ")

    regr = DecisionTreeRegressor(max_depth=int(max_depth))
    regr.fit(X_train, y_train)
    
    
    
    feature_names = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures","famrel","freetime",
                     "goout","Dalc","Walc","health","absences","G1","G2","isMale","isGabrielPereira","isUrban","isLessOrEqual","isLivingTogether","Mjob_cat",
                     "Fjob_cat","reason_cat","guardian_cat",
                     "isExtraEducationalSupport","isFamilyEducationalSupport","isExtraPaidClasses","isExtraCurricularActivities","isAttendedNurserySchool",
                     "wantsHigheEducation","internetAccessAtHome","withRomanticRelationship"]
    
    
    # Save PDF with Tree
    dot_data = tree.export_graphviz(regr, out_file=None, 
                                    feature_names=feature_names, filled=True, 
                                    rounded=True, special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("StudentPerformance-tree")
    print("\nSaved PDF with tree, see the folder\n")
    
    # Predict
    y_pred = regr.predict(X_test)
    
    
    
    
    # Plot image
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Ground Truth vs Predicted")
    plt.show()
    fig.savefig('StudentPerformance-plot.jpg')
    
    
    #Results
    print("Predicted: ")
    print(y_pred)
    print("Actual: ")
    print(y_test)
    
    

    # Compute and print the accuracy
    accuracy = r2_score(y_test, y_pred)
    
    print("Student Performance - Accuracy: {}".format(str(accuracy*100) + "%"))


def startCarEvaluation():
    
    names = ["buying","maint", "doors", "persons", "lug_boot", "safety", "class"]
  
    data = pd.read_csv("car.data.csv", header=None,
                   names=names)
    
    data.buying = data.buying.astype('category')
    data["buying_cat"] = data.buying.cat.codes
    
    data.maint = data.maint.astype('category')
    data["maint_cat"] = data.maint.cat.codes
    
    data.lug_boot = data.lug_boot.astype('category')
    data["lug_boot_cat"] = data.lug_boot.cat.codes
    
    data.safety = data.safety.astype('category')
    data["safety_cat"] = data.safety.cat.codes
    
    cleanup_nums = {"doors":     {"5more": 5},
                     "persons":     {"more": 5}}
    data.replace(cleanup_nums, inplace=True)
    
    data = data.drop(columns=["buying","maint", "lug_boot", "safety"])
    

    X = data.drop(columns=["class"]).values
    y = data["class"].values.flatten()
    
    print("\nWhat percentage would you like to use for your test data?")
    test_option = input("For example enter 10 for 10%, 30 for 30% and so on. Percentage: ")

    # Shuffle datasets, and split into train and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(test_option)/100)
    
    print("\nWhat is the max-depth of the tree?")
    max_depth = input("max-depth: ")

    clf = tree.DecisionTreeClassifier(max_depth=int(max_depth))
    clf.fit(X_train, y_train)
    
    
    feature_names = ["doors","persons","buying_cat", "maint_cat", "lug_boot_cat", "safety_cat"]
    
    target_names = ["unacc","acc","good","vgood"]
    
    # Save PDF with Tree
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=feature_names,
                                    class_names=target_names, filled=True, 
                                    rounded=True, special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("CarEvaluation-tree")
    print("\nSaved PDF with tree, see the folder\n")



    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    
    #Results
    print("Predicted: ")
    print(y_pred)
    print("Actual: ")
    print(y_test)

    # Compute and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Car Evaluation - Accuracy: {}".format(str(accuracy*100) + "%"))
    
    

def startDataset():
    print("What data set would you like to use?")
    option = input("Enter 1 for Iris, 2 for Automobile MPG, 3 for Student Performance: , and 4 for Car Evaluation: ")
    
    if option == str(1):
        startIris()
    if option == str(2):
        startAutomobileMPG()
    if option == str(3):
        startStudentPerformance()
    if option == str(4):
        startCarEvaluation()

    again = input("\nWould you like to start again? (y/n): ")
    if again == "y":
        startDataset()
        
pd.options.display.max_columns = 40
print("Welcome to Decision Tree Classifier")
startDataset()