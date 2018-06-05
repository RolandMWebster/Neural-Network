
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
