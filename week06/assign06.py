
"""
@author: Vitali Karmanov

"""
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

pd.options.display.max_columns = 40


print("############################# Contraceptive dataset ############################")
columns = ['wife_age', 'wife_education', 'husband_education', 'num_children', 'is_islam',
                'wife_not_working', 'husband_occupation', 'living_index', 'media_exp_not_good',
                'contraceptive_method']

data = pd.read_csv('cmc.data', names=columns, sep=",")


X = data.drop(columns=['contraceptive_method'])
y = data[['contraceptive_method']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter=1000)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print("********************** Normal **********************")
print(classification_report(y_test,predictions))



# Two different targets

# Target one

X = data.drop(columns=['media_exp_not_good', 'contraceptive_method'])
y_1 = data[['media_exp_not_good']]


X_train, X_test, y_train, y_test = train_test_split(X, y_1, test_size=0.30)

scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPRegressor(hidden_layer_sizes=(20,20,20), max_iter=1000)
mlp.fit(X_train,y_train)


predictions = mlp.predict(X_test)
predictions = (predictions > 0.5)
print("********************** Target One **********************")
print(classification_report(y_test,predictions))

# Target two

y_2 = data[['contraceptive_method']]

X_train, X_test, y_train, y_test = train_test_split(X, y_2, test_size=0.30)

scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPRegressor(hidden_layer_sizes=(20,20,20), max_iter=1000)
mlp.fit(X_train,y_train)


predictions = mlp.predict(X_test)
predictions = (predictions > 0.5)
print("********************** Target Two **********************")
print(classification_report(y_test,predictions))




print("############################# House votes dataset ############################")     
dataset = pd.read_csv("./house-votes-84.csv", header=None, true_values='y', false_values='n', na_values='?')
dataset.dropna(axis=0, inplace=True, how='any') 
X = dataset.drop(columns=[0]).astype(float)
y = dataset[[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(max_iter=500)
mlp.fit(X_train,y_train.values.ravel())


predictions = mlp.predict(X_test)

print(classification_report(y_test,predictions))


print("############################# Parkinson dataset ############################")
dataset = pd.read_csv("./parkinsons.csv", na_values='?')


dataset = dataset.drop(columns=["name"])

X = dataset.drop(columns=["status"]).values

y = dataset["status"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

# data normalized and transformed
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(22,22,22),max_iter=11500)

clf.fit(X, y)

predictions = clf.predict(X_test)

print(classification_report(y_test,predictions))




print("############################# Student dataset ############################")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=10000)
mlp.fit(X_train,y_train.values.ravel())

predictions = mlp.predict(X_test)


print(classification_report(y_test,predictions))