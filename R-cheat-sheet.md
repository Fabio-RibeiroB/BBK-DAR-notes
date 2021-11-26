## clean environment
```rm(list = ls())``

## Clear console
control + L

## Basic stats
```
mean()
var()
```
## For loops

## Error rate
```mean(y.predict != y.test)```

## Read csv
```read.csv(filename)```

### Create a numeric variable
```as.numeric()```

## Predict: create a vector of predictions
```predict(model, newdata=x.test)```
## Prediction:
```
library(ROCR)
prediction(predictions vector, true class labels vector)
```
### Replications: Returns a vector or a list of the number of replicates for each term in the formula.
```rep(1,100)```
One hundred repeats of 1 in a vector

## Sample: sample takes a sample of the specified size from the elements of x using either with or without replacement.
```sample(x, size, replace = FALSE)```


## K-fold CV (load caret library)
 ```library(caret)```
```createFolds(y=training_data[, target-column], k=number-of-folds)```

# Logistic Regression
```glm(y ~ x, data, family=binomial```
 
y is the target, x is the predictor/s

Useful notation:

Response is dependpent on more than one predictor ```z ~ x + y```
Response is dependent on all predictors (.), except itself (-) ```y ~ .-y``` 

## ifelse: conditional element selection
### If probability is > 0.5, assign 1, else 0
```glm.pred <- ifelse(glm.probs > 0.5, 1, 0)``` 
