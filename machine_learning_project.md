# Classification of Exercise Data

The aim of the project is to determine the manner in which the exercise has been done (coded as `A,B,C,D,E`, so there are five categories) based on the remaining available variables.

## Cleaning the data

Firstly we load the whole data set:


```r
data.train <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
```

Looking at the data we can see that for many of the columns it will not be possible to include them to a model, since they contain only (or mostly) NA values. Also, we exclude people's names and time stamps.

This visual inspection leads to the following choice of predictors:


```r
# STEP 1: all the variables
vars <- names(data.train) 

# STEP 2: selecting those to be used in modelling

# (a) those corresponing to x,y,z & total_accel
vars.2a <- c(vars[grepl("_x",vars)],vars[grepl("_y",vars)],vars[grepl("_z",vars)],
             vars[grepl("total_accel",vars)]);

# (b) combinations of "belt", "arm", "forearm", "dumbbell"
# with "roll", "pitch, "yaw""
aux <- c("belt", "arm", "forearm", "dumbbell");
vars.2b <- c(paste("roll_",aux,sep=""), paste("pitch_",aux,sep=""), paste("yaw_",aux,sep=""))

# adding (a) and (b) together:
vars.2 <- c(vars.2a,vars.2b)

# excluding those which are NA's
vars.2 <- vars.2[!grepl("kurtosis",vars.2) & !grepl("skewness",vars.2) & !grepl("max",vars.2) 
                 & !grepl("min",vars.2) & !grepl("amplitude",vars.2) & !grepl("avg",vars.2) 
                 & !grepl("stddev",vars.2) & !grepl("var",vars.2)]

# STEP 3: to the final set for traning data we add "classe"
vars.3 <- c(vars.2,"classe")
# and take the corresponding data
data.train <- data.train[,vars.3]
```

## Building the random forest model



```r
# Loading packages
library(caret)
library(randomForest)
```

Firstly we create the partition of the data for training and testing. We take 60 percent of the data to the training part.


```r
set.seed(42) # for reproducibility

inTrain <- createDataPartition(y=data.train$classe, p=0.6, list=FALSE)
training <- data.train[inTrain,]
testing <- data.train[-inTrain,]
```

We are going to use **random forests** with all the predictors selected in the cleaning data part, and default parameters of the function from `randomForest` package (it turned out to be much faster than a similar function in `caret` package).


```r
modFit <- randomForest(classe~ ., data=training)
```

Here we show the resulting confusion matrix:


```r
modFit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.62%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3344    3    0    0    1 0.001194743
## B    8 2264    7    0    0 0.006581834
## C    0   14 2035    5    0 0.009250243
## D    0    0   27 1901    2 0.015025907
## E    0    0    0    6 2159 0.002771363
```


Now, we use the model to estimate the testing part of the data:

```r
pred <- predict(modFit,testing); 
testing$predRight <- pred==testing$classe
table(pred,testing$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 2231    7    0    0    0
##    B    1 1511    9    0    0
##    C    0    0 1356   13    0
##    D    0    0    3 1272    9
##    E    0    0    0    1 1433
```

The proportion of correctly classified data is 
99.45195 percent.

## Cross validation and estimating the out of sample error

To estimate the out of sample error, we do the cross validation. We split the data set into training and testing part, estimate the model using the training data and compute its accuracy on the testing part of the data. We keep the proportion of correctly classified data in each of the simulations:


```r
N <- 50         # number of simulations
res=rep(NA,N)   # vector for storing results

for (i in 1:N) {
  inTrain <- createDataPartition(y=data.train$classe, p=0.6, list=FALSE)
  
  training <- data.train[inTrain,]
  testing <- data.train[-inTrain,]
  
  modFit <- randomForest(classe~ ., data=training)
  
  pred <- predict(modFit,testing); 
  testing$predRight <- pred==testing$classe
  
  res[i] <- 100*sum(testing$predRight)/(dim(testing)[1])
}
```

We compute moving averages (average accuracy from the first simulation, first two simulations, first three simulations, etc.). They seem to stabilize, so we take the final value (denoted by a blue line) as our estimate of accuracy.


```r
avg <- cumsum(res)/(1:N)

plot(avg, xlab=c("number of runs"), ylab=c("estimate of the accuracy"))
lines(avg)
lines(1:N, rep( mean(res), N), col=c("blue"))
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9.png) 

So the estimate of the out of sample accuracy is 99.4075962 percent and the estimate of the out of sample error is 0.5924038 percent.

## Realized out of sample error for the submission data

The realized out of sample accuracy turned out to be 100 percent; all 20 observations were classified correctly. This agrees with the earlier computations, since according to them, the expected number of incorrectly classified observations is 0.1184808.
