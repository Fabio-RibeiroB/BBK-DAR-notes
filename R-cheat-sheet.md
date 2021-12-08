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
sappy(my.data, sd) # apply a function to a vector
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
head(my.data)
View(my.data)
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
```createFolds(y=training.data[, target-column], k=number-of-folds)```

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

auc.value.alcohol<-as.numeric() # auc vector
set.seed(11) # Set the value of random seed.

folds <- createFolds(y=training.data[,3],k=10) # Allocate training instances into different folds.
str(folds)

# Conduct 10-fold CV.
for(i in 1:10){
    fold.cv.test <- training.data[folds[[i]],] 
    fold.cv.train <- training.data[-folds[[i]],] 
    trained.model.alcohol <- glm(quality  ~ alcohol, data = fold.cv.train, family = binomial)

    pred.prob.alcohol <- predict(trained.model.alcohol, fold.cv.test, type='response')
    pr.alcohol <- prediction(pred.prob.alcohol, fold.cv.test$quality)

    auroc.alcohol <- performance(pr.alcohol, measure = "auc")
    auroc.alcohol <- auroc.alcohol@y.values[[1]]
    
    auc.value.alcohol<- append(auc.value.alcohol,auroc.alcohol) 
}
print(mean(auc.value.alcohol))
```

## Performance(prediction.obj, measure, x.measure) for evaluations
```
# Calculate TPR and FPR values obtained by different models.
prf.trained.model.1 <- performance(pr.trained.model.1, measure = "tpr", x.measure = "fpr")

# Draw ROC curves for different models.
plot(prf.trained.model.1, col="blue")
```
# Decision Tree
```library(tree)```

# K-Means Clustering
### Uses un-labeled data (unsupervised)
Set seed and use
```km.out <- kmeans(data, k clusters, number of random starting sets)```

Get total within-cluster sum of squares

```km.out$tot.withinss```

Write a for-loop to record tot.withinss when k is 1 to 15. Plot the results and choose the best k (number of clusters).

```
toWithinSS <- rep(0, 15)

for(i in 1:15){
  set.seed(70)
  km.out <- kmeans(dat, i, nstart = 100)
  toWithinSS[i] <- km.out$tot.withinss


}

plot(1:15, toWithinSS, xlab = 'K', ylab = 'Total Within-Cluster SS')
```

Where the gradient starts to flatten is a good choice for k (Elbow method) i.e. lowest number of clusters with a low ```tot.withinss```.

Plot the clusters

```
set.seed(70)

colour.vector <- c('orangered', 'seagreen', 'slateblue', 'magenta', 'sandybrown', 'cyan', 'tomato')

# k-means can produce different results so set a seed
km.out.k6 <- kmeans(dat, 6, nstart =100)
k.c <- km.out.k6$cluster # the vector with the cluster number each point belongs to

plot(dat,
     col = colour.vector[k.c],
     main = 'K6',
     pch = 20,
     cex =2)
```

# Hierarchical Clustering
### Unsupervised, clusters similar points together based on different metrics

Make sure features are columns and transpose otherwise ```t()```.

Use ```hclust(d, method)``` where ```d``` is a dissimilarity structure as produced by ```dist``` i.e. a structure indicating how dissiilar each point is to another. ```method``` is the agglomeration method to be used e.g. ```method = 'single'``` groups together sub-clusters based on closest pairs.

In this example we use correlation to indicate the dissimilarity between variables. But correlation can be negative (and distances cannot) so shift. Use ```as.dist()``` to covert a symmetric square matrix into a ```dist``` structure (which is a triangular matrix).

```
D <- as.dist(1 - cor(t(DF))) # 1 minus to get correlation between 0 to 2
```

Apply hierarchical clustering to the samples using correlation based distance for the Complete Linkage.

```
hclust.cor.comp <- hclust(D, method = 'complete')
plot(hclust.cor.comp, main = 'Complete Linkage', cex = 0.9)
```

Use ```cutree(tree, desired number of groups``` to cut the dendrogram tree made by ```hclust``` to get the desired number k clusters.

If you know the actual groups the data belongs to, you can create a confusion matrix with redicted and true result axes.
```
print(table(predicted = cutree(hclust.obj, k = 2), truth=c(rep(0,20), rep(1,20))))
```
# Principal Component Analysis
### Unsupervised approach for reducing dimensionality or data visualisation.

Perform PCA with ```prcomp(data, scale)```. Set ```scale=True``` so that the variables all have zero mean and std of 1 (standardised). This way we don't bias particular variables as being more important because of different units.

```
pr.out <- prcomp(USArrests, scale=T)

pr.out$center # The mean of the variables become the centre of the data i.e. the origin becomes the centroid

pr.out$scale # Scaling factor to normalise each variable

pr.out$rotation # Table showing the component of each attribute along each of the principal components

```

Use biplot to show the plot the data against the first two PCs (left and bottom axes) and standard deviations (top and right axes).

```biplot(pr.out, scale = TRUE)```

Since PC1 is the most important new feature, it will explain most of the variance. Plot the proportion of variance explained by each component.

```
pr.var <- pr.out$sdev^2 # variance
pve <- pr.var/sum(pr.var) # proportion

plot(pve,
xlab = " Principal Component", ylab = " Proportion of Variance Explained",
ylim=c(0,1), type='b')

plot(cumsum(pve),
xlab = " Principal Component ", ylab =" Cumulative Proportion of Variance Explained",
ylim = c(0,1), type = 'b')
```
