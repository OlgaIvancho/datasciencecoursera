library(ggplot2)
library(caret)

## Loading the package: lattice
library(randomForest)

## randomForest 4.6-10

library(e1071)
library(gbm)

## Loading required package: survival
## Loading required package: splines

## Attaching package: 'survival'

## The following object is masked from 'package:caret':
##     cluster

## Loading required package: parallel
## Loaded gbm 2.1
library(doParallel)
## Loading required package: foreach
## Loading required package: iterators
library(survival)
library(splines)
library(plyr)
setwd("~/GitHub/Machine Learning PRJ1")
training <- read.csv("~/GitHub/PracMacLearn/data/pml-training.csv", na.strings=c("#DIV/0!"), row.names = 1)
testing <- read.csv("~/GitHub/PracMacLearn/data/pml-testing.csv", na.strings=c("#DIV/0!"), row.names = 1)
training <- training[, 6:dim(training)[2]]

treshold <- dim(training)[1] * 0.95
#Remove columns with more than 95% of NA or "" values
goodColumns <- !apply(training, 2, function(x) sum(is.na(x)) > treshold  || sum(x=="") > treshold)

training <- training[, goodColumns]

badColumns <- nearZeroVar(training, saveMetrics = TRUE)

training <- training[, badColumns$nzv==FALSE]

training$classe = factor(training$classe)

#Partition rows into training and crossvalidation

inTrain <- createDataPartition(training$classe, p = 0.6)[[1]]
crossv <- training[-inTrain,]
training <- training[ inTrain,]
inTrain <- createDataPartition(crossv$classe, p = 0.75)[[1]]
crossv_test <- crossv[ -inTrain,]
crossv <- crossv[inTrain,]


testing <- testing[, 6:dim(testing)[2]]
testing <- testing[, goodColumns]
testing$classe <- NA
testing <- testing[, badColumns$nzv==FALSE]

mod1 <- train(classe ~ ., data=training, method="rf")

pred1 <- predict(mod1, crossv)

confusionMatrix(pred1, crossv$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    3    0    0    0
##          B    1 1135    6    0    0
##          C    0    1 1020    4    0
##          D    0    0    0  960    1
##          E    1    0    0    1 1081
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.995, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.996    0.994    0.995    0.999
## Specificity             0.999    0.999    0.999    1.000    1.000
## Pos Pred Value          0.998    0.994    0.995    0.999    0.998
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.163    0.184
## Detection Prevalence    0.285    0.194    0.174    0.163    0.184
## Balanced Accuracy       0.999    0.998    0.997    0.997    0.999
