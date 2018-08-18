
# Info --------------------------------------------------------------------

# Building a neural network to classify hand written numbers 1-9


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

print(Sys.time() - start.time)

