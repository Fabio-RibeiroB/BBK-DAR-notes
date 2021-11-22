# My DAR notes
Data Analytics with R notes

##  Week 1
### **Big Data and Data Analytics**
#### Lecture
Big data spans four "dimenions":
* Velocity (rate of flow of data, e.g. real-time/periodic)
* Volume (size in bytes)
* Veracity (uncertainty of data, e.g.  noise, biases)
* Variety (type of data, e.g. vidio, table)

This module focuses on **Statistical Learning**.

Statistical Learning is all about how to estimate the true function f underlying the relationshi between input and output variables.

We estimate f using training data and a statistical learning method.

#### Labs
Create a vector with combine function and assign to variable
```
u <- c(0,1,2)
```
Vector from -0.1, +0.5 increment, 11 terms
```
v <- seq(-0.1, by=0.5, length=11)
```
Add and muptiply to every term in the vector

```
u <- u + 1
v< <- v*0.8
```
Combine u and v to make w that has terms form u then v
```
w<-c(u, v)
w[1] # is the first term
w[14:16] # 14-16th value
w[c(2, 5, 9, 21)] #2nd, 5th…
```
Create the matrix

1   3  5   7    9
11 13 15 17 19
21  23  25  27  29
31  33  35  37  39

```

b_matrix = matrix(seq(1, 39, by=2), 4, 5, byrow=TRUE) # byrow=True, the matrix is filed by rows

```

Getting a sub-matrix
First two rows
First two columns
Get rows 1 and 3, and columns 2 and 4

```
A[1:2,]
A[,1:2]
A[c(1,3), c(2,4)]

```

Subracting the first and third row
Combining vectors (x,y,z) to make amatrix where each column is vector

```
A[-c[1,3],]
A <- cbind(x,y,z) # cbind to combine by columns rbind to combine by rows
```

Renaming the rows and print
```
A = matrix(c(x, y, z), 3, 3, dimnames = list(c("a", "b", "c"), c("a", "b", "c")))
Rownames(A) <- c("row1", …)
Colnames <- c("col1", …)
print(A)
```

## Week 2
### **Basic Stats**
#### Lecutres
Insert picture summary of maths



