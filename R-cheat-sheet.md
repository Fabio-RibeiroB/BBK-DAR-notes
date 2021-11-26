## clean environment
```rm(list = ls())``

## Maximise R scipt
control + shift + 1

## Clear console
control + L

## Basic stats and other useful functions
```
mean()
var(x)
sd(x)
sappy(my_data, sd) # apply a function to a vector
cov(x,y)
cor(x,y)
length(x)
sum(x)
median(x)
sort(x)
table(
summary(x)
```
## For loops

## Error rate
```mean(y.predict != y.test)```

## Read csv
```read.csv(filename)```
```
head(my_data)
View(my_data)
```
### Create a numeric variable or factor
```
as.numeric()
as.factor()
```


# Linear Regression
```lm.fit <- lm(y ~ x, data)```
```
plot(x, y, 
     xlab, ylab, 
     main = "Title")
abline(lm.fit)
```
```summary(lm.fit)```

## Predict: create a vector of predictions
```predict(model, newdata=x.test)```

## Prediction:
```
library(ROCR)
prediction(predict vector, true class labels vector)
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

# K-fold Cross-Validation
 
Conduct 10-fold cross validation to obtain the average AUROC value by using alcohol as the only feature.
```
# Use package "caret" for running the k-fold CV.
library(caret)
library(ROCR)

auc_value_alcohol<-as.numeric() # auc vector
set.seed(11) # Set the value of random seed.

folds <- createFolds(y=training_data[,3],k=10) # Allocate training instances into different folds.
str(folds)

# Conduct 10-fold CV.
for(i in 1:10){
    fold_cv_test <- training_data[folds[[i]],] 
    fold_cv_train <- training_data[-folds[[i]],] 
    trained_model_alcohol <- glm(quality  ~ alcohol, data = fold_cv_train, family = binomial)

    pred_prob_alcohol <- predict(trained_model_alcohol, fold_cv_test, type='response')
    pr_alcohol <- prediction(pred_prob_alcohol, fold_cv_test$quality)

    auroc_alcohol <- performance(pr_alcohol, measure = "auc")
    auroc_alcohol <- auroc_alcohol@y.values[[1]]
    
    auc_value_alcohol<- append(auc_value_alcohol,auroc_alcohol) 
}
print(mean(auc_value_alcohol))
```

## Performance(prediction.obj, measure, x.measure) for evaluations
```
# Calculate TPR and FPR values obtained by different models.
prf_trained_model_1 <- performance(pr_trained_model_1, measure = "tpr", x.measure = "fpr")

# Draw ROC curves for different models.
plot(prf_trained_model_1, col="blue")
```
# Decision Tree
```library(tree)```  
