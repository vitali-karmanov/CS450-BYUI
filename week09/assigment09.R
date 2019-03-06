install.packages('e1071', dependencies = TRUE)

library (e1071)

# Read dataset
vowel <- read.csv("vowel.csv")

# Split in test and training
allRows <- 1:nrow(vowel)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

# The test set contains all the test rows
vowelTest <- vowel[testRows,]

# The training set contains all the other rows
vowelTrain <- vowel[-testRows,]

# declare variables
bestAccuracy <- 0.0
bestParam <- 0.0
params <- c(0.00001,.0001, .001, .01, .1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
param <- params[1]

# Loop through parameters
for (param in params) {
  
  print(paste("Gamma parameter:", param))
  
  # Create model
  model <- svm(Class~., data = vowelTrain, kernel = "radial", gamma = param, cost = 1)
  
  # Delete the target column
  prediction <- predict(model, vowelTest[,-13])
  
  confusionMatrix <- table(pred = prediction, true = vowelTest$Class)
  
  # Calculate Accuracy
  agreement <- prediction == vowelTest$Class
  accuracy <- prop.table(table(agreement))
  
  if (accuracy[2] >= bestAccuracy) {
    bestParam <- param
    bestAccuracy <- accuracy[2]
  }
}


print(confusionMatrix)

print(paste("Best accuracy:",bestAccuracy))
print(paste("Best gamma parameter:",bestParam))

plot(confusionMatrix)


#################letters#################

# Read dataset
letters <- read.csv("letters.csv")

allRows <- 1:nrow(letters)
testRows <- sample(allRows, trunc(length(allRows) * 0.2))

# The test set contains all the test rows
lettersTest <- letters[testRows,]

# The training set contains all the other rows
lettersTrain <- letters[-testRows,]


# declare variables
bestAccuracy <- 0.0
bestParam <- 0.0
params <- c(0.00001,.0001, .001, .01, .1, 1.0, 10.0, 100.0, 1000.0, 10000.0)

param <- params[1]

# Loop through parameters
for (param in params) {
  
  print(paste("Gamma parameter:", param))
  
  model <- svm(letter~., data = lettersTrain, kernel = "radial", gamma = param, cost = 1)
  
  prediction <- predict(model, lettersTest[,-1])
  
  confusionMatrix <- table(pred = prediction, true = lettersTest$letter)
  
  agreement <- prediction == lettersTest$letter
  accuracy <- prop.table(table(agreement))
  
  if (accuracy[2] >= bestAccuracy) {
    bestParam <- param
    bestAccuracy <- accuracy[2]
  }
}


print(confusionMatrix)
print(paste("Best accuracy:",bestAccuracy))
print(paste("Best gamma parameter:",bestParam))

plot(confusionMatrix)


