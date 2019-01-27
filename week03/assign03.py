# -*- coding: utf-8 -*-
"""
@author: Vitali Karmanov
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("\nHow many neighbors would you like to use?")
    neighbors_option = input("Neighbors: ")
    
    # Create the classifier
    classifier = KNeighborsClassifier(n_neighbors=int(neighbors_option))

    # Train the classifier
    classifier.fit(X_train, y_train)

    
    # Make predictions on the test data
    y_pred = classifier.predict(X_test)

    # Compute and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Car Evaluation - Accuracy: {}".format(str(accuracy*100) + "%"))
    
    
    
def startAutomobileMPG():
    
    names = ["mpg","cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year","origin","car_name"]
  
    data = pd.read_csv("auto-mpg.data.csv", header=None,
                   names=names, na_values=["?"], delim_whitespace=True)
    
    data.horsepower = data.horsepower.fillna(data.horsepower.median())
    
    data = data.drop(columns=["car_name"])
    
    X = data.drop(columns=["mpg"]).values
    y = data["mpg"].values.flatten()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print("\nHow many neighbors would you like to use?")
    neighbors_option = input("Neighbors: ")

    regr = KNeighborsRegressor(n_neighbors=int(neighbors_option))
    regr.fit(X_train, y_train)
    
    predictions = regr.predict(X_test)

    # Compute and print the accuracy
    accuracy = r2_score(y_test, predictions)
    
    print("Automobile MPG - Accuracy: {}".format(str(accuracy*100) + "%"))
    


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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("\nHow many neighbors would you like to use?")
    neighbors_option = input("Neighbors: ")

    regr = KNeighborsRegressor(n_neighbors=int(neighbors_option))
    regr.fit(X_train, y_train)
    
    predictions = regr.predict(X_test)

    # Compute and print the accuracy
    accuracy = r2_score(y_test, predictions)
    
    print("Student Performance - Accuracy: {}".format(str(accuracy*100) + "%"))
   

def startDataset():
    print("What data set would you like to use?")
    option = input("Enter 1 for Car Evaluation, or 2 for Automobile MPG, or 3 for StudentPerformance: ")
    
    if option == str(1):
        startCarEvaluation()
    if option == str(2):
        startAutomobileMPG()
    if option == str(3):
        startStudentPerformance()

    again = input("\nWould you like to start again? (y/n): ")
    if again == "y":
        startDataset()
        
pd.options.display.max_columns = 20
print("Welcome to Census Data Set example")
startDataset()