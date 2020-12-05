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
