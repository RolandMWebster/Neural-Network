# Packages ----------------------------------------------------------------

library(tidyr)
library(plyr)
library(dplyr)
library(ggplot2)

# Read in Data ------------------------------------------------------------

train.data <- read.csv("mnist_train.csv",
                       stringsAsFactors = FALSE, 
                       header = FALSE)

train.data <- train.data[sample.int(nrow(train.data),
                                    replace = FALSE),]

# Data arranged as row 1 col 1, row 1 col 2, ...
kTrainObs <- 50000
kImageWidth <- 28
kImageHeight <- 28
kNeurons <- 30
kOutputs <- 10

# Creating Model Input ----------------------------------------------------

train.input <- train.data[(1:kTrainObs),-1]

train.input <- train.input / max(train.input)

train.input <- t(train.input)

train.labels <- train.data[(1:kTrainObs),1]

skeleton <- array(data = rep(1,10),
                  dim = c(1,10))

train.labels <- train.labels %*% skeleton

for(i in 1:ncol(train.labels)){
  for(j in 1:nrow(train.labels)){
    if(train.labels[j,i] == i-1){
      train.labels[j,i] <- 1}else{
        train.labels[j,i] <- 0
      }
  }
}

train.labels <- t(train.labels)


# Creating Test Data ------------------------------------------------------

test.input <- train.data[(kTrainObs+1):nrow(train.data),-1]

test.input <- test.input / max(test.input)

test.input <- t(test.input)

test.labels <- train.data[(kTrainObs+1):nrow(train.data),1]

skeleton <- array(data = rep(1,10),
                  dim = c(1,10))

test.labels <- test.labels %*% skeleton

for(i in 1:ncol(test.labels)){
  for(j in 1:nrow(test.labels)){
    if(test.labels[j,i] == i-1){
      test.labels[j,i] <- 1}else{
        test.labels[j,i] <- 0
      }
  }
}

test.labels <- t(test.labels)

















