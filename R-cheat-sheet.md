

## Read csv
```read.csv(filename)```

## Create a numeric variable
```as.numeric()```

## K-fold CV
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
