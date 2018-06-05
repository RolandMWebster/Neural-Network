Neural Network Results
================
Roland Webster
5 June 2018

Outline
-------

Train the model and analyse performance.

Data Prep
---------

``` r
# Packages ----------------------------------------------------------------

library(tidyr)
```

    ## Warning: package 'tidyr' was built under R version 3.4.1

``` r
library(plyr)
library(dplyr)
```

    ## Warning: package 'dplyr' was built under R version 3.4.1

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:plyr':
    ## 
    ##     arrange, count, desc, failwith, id, mutate, rename, summarise,
    ##     summarize

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(ggplot2)
```

    ## Warning: package 'ggplot2' was built under R version 3.4.1

``` r
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
kTrainObs <- 50000


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
```

User Defined Functions
----------------------

``` r
# Define Sigmoid Function -------------------------------------------------

# Define our sigmoid function. This is used to map neuron values onto (0,1)

sigmoid <- function(x){
  1/(1 + exp(-x))
}

# Define our first derivative of the sigmoid function. This is used when calculating the error.
diff_sigmoid <- function(x){
  sigmoid(x) * (1 - sigmoid(x))
}
```

Train Model
-----------

``` r
# Model Parameters --------------------------------------------------------

kNetworkShape <- c(784,30,10) # The shape of the model (including input and output layers).
kBatchSize <- 10 # The number of observations passed per batch.
kTrainingRate <- 3 # The training rate for the model.
kEpochs <- 30 # Number of times the full data set is passed through the model.

kNetworkLength <- length(kNetworkShape) # Total number of layers in the model.

# Skeleton List -----------------------------------------------------------

# Create appropriately sized list for the weights, biases and neurons.
# This list should have the same number of elements as there are layers in the model.
list.skeleton <- as.list(kNetworkShape)

# Rename each element of weights. These names correspond to the position of the layer.
# in the model.
names(list.skeleton) <- c(1:kNetworkLength)

# Change the value of each element in our list to match its name.
# This allows use to use as simple lapply call to build our lists.
for(i in 1:kNetworkLength){
list.skeleton[i] <- as.numeric(names(list.skeleton[i]))
}


# Weights -----------------------------------------------------------------

# Start by randomly generating values from a standard normal distribution.
weights <- lapply(list.skeleton,
                function(x){
                  x <- array(data = rnorm(n = kNetworkShape[x]*kNetworkShape[x-1],
                                          0,
                                          1),
                             dim = c(kNetworkShape[x],
                                     kNetworkShape[x-1]))
                })


# Biases ------------------------------------------------------------------

# Start by randomly generating values from a standard normal distribution.
biases <- lapply(list.skeleton,
               function(x){
                 x <- array(data = rnorm(n = kNetworkShape[x],
                                         0,
                                         1),
                            dim = c(kNetworkShape[x],
                                    kBatchSize)) # We duplicate our bias vectors so we can multiply with each observation of our batch.
               })



# Activation of Neurons ---------------------------------------------------

a.neurons <- lapply(list.skeleton,
                  function(x){
                    x <- array(data = c(0), # Fill with 0s for now.
                               dim = c(kNetworkShape[x],
                                       kBatchSize))
                  })




# Weighted Activation of Neurons ------------------------------------------

z.neurons <- lapply(list.skeleton,
                  function(x){
                    x <- array(data = c(0), # Fill with 0s for now.
                               dim = c(kNetworkShape[x],
                                       kBatchSize))
                  })


# Errors ------------------------------------------------------------------

errors <- lapply(list.skeleton,
                  function(x){
                    x <- array(data = c(0), # Fill with 0s for now.
                               dim = c(kNetworkShape[x],
                                       kBatchSize))
                  })


# Start Time Log ----------------------------------------------------------

start.time <- Sys.time()

# Start of Epoch Loop -----------------------------------------------------
for(epoch in 1:kEpochs){

# Shuffle our data and create our batches
sample <- sample.int(kTrainObs,
                   replace = FALSE)

shuffled.data <- train.input[,sample]
shuffled.labels <- train.labels[,sample]


# Reset Correct Predictions Counter ---------------------------------------

correctly.predicted <- 0


# Start of Batch Loop -----------------------------------------------------
for(batchNo in 1:(kTrainObs/kBatchSize)){

# Assign our input values given our batches

a.neurons[[1]] <- shuffled.data[,(((batchNo-1)*kBatchSize) + 1):(kBatchSize*batchNo)]

input.labels <- shuffled.labels[,(((batchNo-1)*kBatchSize) + 1):(kBatchSize*batchNo)]



# Calculate Activation and Weighted Activation of Neurons -----------------

# Feedforward  
for(i in 2:kNetworkLength){

z.neurons[[i]] <- (weights[[i]] %*% a.neurons[[i-1]]) + biases[[i]] 
a.neurons[[i]] <- sigmoid(z.neurons[[i]])

}


# Store Results -----------------------------------------------------------

predictions <- sapply(as.data.frame(a.neurons[[kNetworkLength]]),
                    function(x){
                      x <- which.max(x)
                      output <- array(data = rep(0,kNetworkShape[kNetworkLength]))
                      output[x] <- 1
                      output
                      })

# Update correctly predicted counter to tell us how well our model is doing.
correctly.predicted <- correctly.predicted + sum(input.labels * predictions)

# Calculate Output Error --------------------------------------------------

errors[[kNetworkLength]] <- (a.neurons[[kNetworkLength]] - input.labels) * diff_sigmoid(z.neurons[[kNetworkLength]])


# Backpropagate the error -------------------------------------------------

for(i in (kNetworkLength - 1):2){

errors[[i]] <- (t(weights[[i+1]]) %*% errors[[i+1]]) * diff_sigmoid(z.neurons[[i]])

}


# Update Weights ----------------------------------------------------------

for(i in kNetworkLength:2){
weights[[i]] <- weights[[i]] - ((kTrainingRate / kBatchSize) * (errors[[i]] %*% t(a.neurons[[i-1]]))) 
biases[[i]] <- biases[[i]] - ((kTrainingRate / kBatchSize) * colSums(errors[[i]]))
}


} # End of batch loop

print(paste0(epoch,
           ": ", 
           correctly.predicted, 
           " / ", 
           kTrainObs,
           " (",
           100*correctly.predicted/kTrainObs,
           "%)"))


} # End of Epoch loop
```

    ## [1] "1: 39502 / 50000 (79.004%)"
    ## [1] "2: 45555 / 50000 (91.11%)"
    ## [1] "3: 46280 / 50000 (92.56%)"
    ## [1] "4: 46658 / 50000 (93.316%)"
    ## [1] "5: 46927 / 50000 (93.854%)"
    ## [1] "6: 47146 / 50000 (94.292%)"
    ## [1] "7: 47312 / 50000 (94.624%)"
    ## [1] "8: 47437 / 50000 (94.874%)"
    ## [1] "9: 47493 / 50000 (94.986%)"
    ## [1] "10: 47628 / 50000 (95.256%)"
    ## [1] "11: 47719 / 50000 (95.438%)"
    ## [1] "12: 47833 / 50000 (95.666%)"
    ## [1] "13: 47937 / 50000 (95.874%)"
    ## [1] "14: 47933 / 50000 (95.866%)"
    ## [1] "15: 47979 / 50000 (95.958%)"
    ## [1] "16: 48062 / 50000 (96.124%)"
    ## [1] "17: 48090 / 50000 (96.18%)"
    ## [1] "18: 48172 / 50000 (96.344%)"
    ## [1] "19: 48226 / 50000 (96.452%)"
    ## [1] "20: 48256 / 50000 (96.512%)"
    ## [1] "21: 48309 / 50000 (96.618%)"
    ## [1] "22: 48319 / 50000 (96.638%)"
    ## [1] "23: 48382 / 50000 (96.764%)"
    ## [1] "24: 48437 / 50000 (96.874%)"
    ## [1] "25: 48442 / 50000 (96.884%)"
    ## [1] "26: 48468 / 50000 (96.936%)"
    ## [1] "27: 48494 / 50000 (96.988%)"
    ## [1] "28: 48507 / 50000 (97.014%)"
    ## [1] "29: 48542 / 50000 (97.084%)"
    ## [1] "30: 48571 / 50000 (97.142%)"

``` r
print(Sys.time() - start.time)
```

    ## Time difference of 4.293096 mins

Test Out of Sample Performance
------------------------------

``` r
# Out of sample testing ---------------------------------------------------
kTestObs <- ncol(test.input)

correctly.predicted.test <- 0


for(batchNo in 1:(kTestObs/kBatchSize)){
  
  # Here we need to assign our input values given our batches
  
  a.neurons[[1]] <- test.input[,(((batchNo-1)*kBatchSize) + 1):(kBatchSize*batchNo)]
  
  input.labels <- test.labels[,(((batchNo-1)*kBatchSize) + 1):(kBatchSize*batchNo)]
  
  
  
  # Calculate Activation and Weighted Activation of Neurons -----------------
  
  # Feedforward  
  for(i in 2:kNetworkLength){
    
    z.neurons[[i]] <- (weights[[i]] %*% a.neurons[[i-1]]) + biases[[i]] 
    a.neurons[[i]] <- sigmoid(z.neurons[[i]])
    
  }
  
  
  # Store Results -----------------------------------------------------------
  
  predictions <- sapply(as.data.frame(a.neurons[[kNetworkLength]]),
                        function(x){
                          x <- which.max(x)
                          output <- array(data = rep(0,kNetworkShape[kNetworkLength]))
                          output[x] <- 1
                          output
                        })
  
  correctly.predicted.test <- correctly.predicted.test + sum(predictions * input.labels)
  
  
}


print(paste0("Correctly classified: ",
            correctly.predicted.test, 
            "/", 
            ncol(test.input),
            " (",
            100*(correctly.predicted.test / ncol(test.input)),
            "%)"))
```

    ## [1] "Correctly classified: 9461/10000 (94.61%)"
