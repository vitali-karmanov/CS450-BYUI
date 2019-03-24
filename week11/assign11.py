# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:34:55 2019

@author: Vitali
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import datasets

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def startVotes():
    
    dataset = pd.read_csv("./house-votes-84.csv", header=None, true_values='y', false_values='n', na_values='?')
    dataset.dropna(axis=0, inplace=True, how='any') 
    
    X = dataset.drop(columns=[0]).astype(float)
    y = dataset[[0]]

    return X, y

def startContraceptive():
    columns = ['wife_age', 'wife_education', 'husband_education', 'num_children', 'is_islam',
                    'wife_not_working', 'husband_occupation', 'living_index', 'media_exp_not_good',
                    'contraceptive_method']
    
    data = pd.read_csv('cmc.data', names=columns, sep=",")

    X = data.drop(columns=['contraceptive_method'])
    y = data[['contraceptive_method']]

    return X, y


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
    
    return X, y

def startIris():

    # Load Iris datasets from sklearn
    iris = datasets.load_iris()
    return iris

def startStudent():
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
    
    X = data.drop(columns=["G3"]).values.astype(float)
    y = data["G3"]
    
    return X, y

print("############################# Car Evaluation ############################")
      
################# KNeighborsClassifier #################
X, y = startCarEvaluation()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the classifier
classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Compute and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("KNeighborsClassifier - Accuracy: {}".format(str(accuracy*100) + "%"))

################# DecisionTreeClassifier #################

X, y = startCarEvaluation()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = tree.DecisionTreeClassifier(max_depth=500)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ("DecisionTreeClassifier - Accuracy: " + str(accuracy_score(y_test,y_pred)*100) + "%")

################# GaussianNBClassifier #################

X, y = startCarEvaluation()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifierGaussian = GaussianNB()
classifierGaussian.fit(X_train, y_train)

# Print the % of accuracy 
print("GaussianNBClassifier - Accuracy: " + str(classifierGaussian.score(X_test, y_test)*100) + "%")

################# BaggingClassifier #################

X, y = startCarEvaluation()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = BaggingClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("BaggingClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

################# AdaBoostClassifier #################

X, y = startCarEvaluation()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = AdaBoostClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("AdaBoostClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

################# RandomForestClassifier #################

X, y = startCarEvaluation()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = RandomForestClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("RandomForestClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

print("\n############################# Iris ############################")
      
################# KNeighborsClassifier #################

X_train, X_test, y_train, y_test = train_test_split(startIris().data, startIris().target, test_size=0.3)

# Create the classifier
classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Compute and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("KNeighborsClassifier - Accuracy: {}".format(str(accuracy*100) + "%"))


################# DecisionTreeClassifier #################

X_train, X_test, y_train, y_test = train_test_split(startIris().data, startIris().target, test_size=0.3)

clf = tree.DecisionTreeClassifier(max_depth=500)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ("DecisionTreeClassifier - Accuracy: " + str(accuracy_score(y_test,y_pred)*100) + "%")

################# GaussianNBClassifier #################

X_train, X_test, y_train, y_test = train_test_split(startIris().data, startIris().target, test_size=0.3)

classifierGaussian = GaussianNB()
classifierGaussian.fit(X_train, y_train)

# Print the % of accuracy 
print("GaussianNBClassifier - Accuracy: " + str(classifierGaussian.score(X_test, y_test)*100) + "%")

################# BaggingClassifier #################

X_train, X_test, y_train, y_test = train_test_split(startIris().data, startIris().target, test_size=0.3)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = BaggingClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("BaggingClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))


################# AdaBoostClassifier #################

X_train, X_test, y_train, y_test = train_test_split(startIris().data, startIris().target, test_size=0.3)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = AdaBoostClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("AdaBoostClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

################# RandomForestClassifier #################

X_train, X_test, y_train, y_test = train_test_split(startIris().data, startIris().target, test_size=0.3)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = RandomForestClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("RandomForestClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best Param: " + str(x))


print("\n############################# Student ############################")
      
################# KNeighborsClassifier #################

X, y = startStudent()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the classifier
classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Compute and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("KNeighborsClassifier - Accuracy: {}".format(str(accuracy*100) + "%"))


################# DecisionTreeClassifier #################

X, y = startStudent()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = tree.DecisionTreeClassifier(max_depth=500)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ("DecisionTreeClassifier - Accuracy: " + str(accuracy_score(y_test,y_pred)*100) + "%")

################# GaussianNBClassifier #################

X, y = startStudent()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifierGaussian = GaussianNB()
classifierGaussian.fit(X_train, y_train)

# Print the % of accuracy 
print("GaussianNBClassifier - Accuracy: " + str(classifierGaussian.score(X_test, y_test)*100) + "%")

################# BaggingClassifier #################

X, y = startStudent()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = BaggingClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("BaggingClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

################# AdaBoostClassifier #################

X, y = startStudent()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = AdaBoostClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x

print ("AdaBoostClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

################# RandomForestClassifier #################

X, y = startStudent()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = RandomForestClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
    

print ("RandomForestClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

print("\n############################# Votes ############################")
      
################# KNeighborsClassifier #################

X, y = startVotes()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

# Create the classifier
classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Compute and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("KNeighborsClassifier - Accuracy: {}".format(str(accuracy*100) + "%"))


################# DecisionTreeClassifier #################

X, y = startVotes()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

clf = tree.DecisionTreeClassifier(max_depth=500)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ("DecisionTreeClassifier - Accuracy: " + str(accuracy_score(y_test,y_pred)*100) + "%")

################# GaussianNBClassifier #################

X, y = startVotes()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

classifierGaussian = GaussianNB()
classifierGaussian.fit(X_train, y_train)

# Print the % of accuracy 
print("GaussianNBClassifier - Accuracy: " + str(classifierGaussian.score(X_test, y_test)*100) + "%")

################# BaggingClassifier #################

X, y = startVotes()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = BaggingClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("BaggingClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

################# AdaBoostClassifier #################

X, y = startVotes()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = AdaBoostClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("AdaBoostClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

################# RandomForestClassifier #################

X, y = startVotes()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = RandomForestClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("RandomForestClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))


print("\n############################# Contraceptive ############################")
      
################# KNeighborsClassifier #################

X, y = startContraceptive()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

# Create the classifier
classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Compute and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("KNeighborsClassifier - Accuracy: {}".format(str(accuracy*100) + "%"))


################# DecisionTreeClassifier #################

X, y = startContraceptive()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

clf = tree.DecisionTreeClassifier(max_depth=500)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ("DecisionTreeClassifier - Accuracy: " + str(accuracy_score(y_test,y_pred)*100) + "%")

################# GaussianNBClassifier #################

X, y = startContraceptive()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

classifierGaussian = GaussianNB()
classifierGaussian.fit(X_train, y_train)

# Print the % of accuracy 
print("GaussianNBClassifier - Accuracy: " + str(classifierGaussian.score(X_test, y_test)*100) + "%")

################# BaggingClassifier #################

X, y = startContraceptive()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = BaggingClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("BaggingClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

################# AdaBoostClassifier #################

X, y = startContraceptive()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = AdaBoostClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("AdaBoostClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))

################# RandomForestClassifier #################

X, y = startContraceptive()

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

best_param = 0
best_accuracy = 0

for x in range(1, 1002, 100):
    clf = RandomForestClassifier(n_estimators=x)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_param = x
        
print ("RandomForestClassifier - Accuracy: " + str(best_accuracy*100) + "% - Best n_estimators: " + str(x))
