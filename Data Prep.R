
# Info --------------------------------------------------------------------

# Data prep code for the model.
# This script is used to load in the raw .csv file and format it ready for the model.
# This includes creating a train/test split of the data.


# Packages ----------------------------------------------------------------

library(tidyr)
library(plyr)
library(dplyr)
library(ggplot2)

# Read in Data ------------------------------------------------------------

train.data <- read.csv("mnist_train.csv",
                       stringsAsFactors = FALSE, 
                       header = FALSE)
# Notes: Each row of train.data is an image. Data is arranged as: row 1 col 1, row 1 col 2, ... 


# Set Seed ----------------------------------------------------------------

set.seed(1111)


# Shuffle Data ------------------------------------------------------------

# We shuffle the data to make we have an appropriate distribution of each number
# in both the training and test data sets.
train.data <- train.data[sample.int(nrow(train.data),
                                    replace = FALSE),]


# Define Train Observations -----------------------------------------------

# For now we will use 50,000 of the 60,000 observations to train our model.
# The remaining 10,000 will be used to test the model performance.
kTrainObs <- 20000


# Creating Model Input ----------------------------------------------------

# Select the columns that contain pixel values for each image. We are deselecting
# the first column which contains the value of the handwritten digit.
train.input <- train.data[(1:kTrainObs),-1]

# Our pixel values range from 0-255, with 0 being solid black and 255 being solid white.
# Standardize our pixel values, mapping them onto [0,1]
# This stops our sigmoid function from geting caught at the limits (where the gradient is
# particularly shallow.)
train.input <- train.input / max(train.input)

# Transpose our data (personal preference)
train.input <- t(train.input)

# Seperate our train labels.
train.labels <- train.data[(1:kTrainObs),1]

# Transform our labels from single values to vectors of length 10.
# Start with an array of dimensions (1,10), populated with 1s.
skeleton <- array(data = rep(1,10),
                  dim = c(1,10))

# Mulitply these to get a vector for each image label.
# Each vector contains the value of the image repeated in each element.
train.labels <- train.labels %*% skeleton

# Let x be an image label and v the corresponding vector
# Then for v_i where i = x, set equal to 1 and set equal to 0 otherwise.
for(i in 1:ncol(train.labels)){
  for(j in 1:nrow(train.labels)){
    if(train.labels[j,i] == i-1){
      train.labels[j,i] <- 1}else{
        train.labels[j,i] <- 0
      }
  }
}

# Transpose labels (personal preference)
train.labels <- t(train.labels)


# Creating Test Data ------------------------------------------------------

# Perform exactly the same for the test data.

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

















