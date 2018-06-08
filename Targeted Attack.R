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

# Activation of Neurons ---------------------------------------------------

a.neurons <- lapply(list.skeleton,
                    function(x){
                      x <- array(data = c(0), # Fill with 0s for now.
                                 dim = c(kNetworkShape[x],
                                         1))
                    })




# Weighted Activation of Neurons ------------------------------------------

z.neurons <- lapply(list.skeleton,
                    function(x){
                      x <- array(data = c(0), # Fill with 0s for now.
                                 dim = c(kNetworkShape[x],
                                         1))
                    })

# Weights -----------------------------------------------------------------

# Start by randomly generating values from a standard normal distribution.
attack.weights <- lapply(list.skeleton,
                        function(x){
                          x <- array(data = rnorm(n = kNetworkShape[x]*kNetworkShape[x-1],
                                                  0,
                                                  1),
                                     dim = c(kNetworkShape[x],
                                             kNetworkShape[x-1]))
                        })


# Biases ------------------------------------------------------------------

# Start by randomly generating values from a standard normal distribution.
attack.biases <- lapply(list.skeleton,
                        function(x){
                          x <- array(data = rnorm(n = kNetworkShape[x],
                                                  0,
                                                  1),
                                     dim = c(kNetworkShape[x],
                                             1)) # We duplicate our bias vectors so we can multiply with each observation of our batch.
                        })


# Assign a[[1]] neuron ----------------------------------------------------

a.neurons[[1]] <- array(data = runif(kNetworkShape[1], 0, 255) / 255,
                        dim = c(kNetworkShape[1],
                                1))

test <- a.neurons[[1]]

input.labels <- train.input[,3]


output.labels <- array(data = c(0,1,0,0,0,0,0,0,0,0),
                      dim = c(kNetworkShape[kNetworkLength],
                              1))

for(batch in 1:1000){

# Feed Forward ------------------------------------------------------------

for(i in 2:kNetworkLength){
  
  z.neurons[[i]] <- (weights[[i]] %*% a.neurons[[i-1]]) + biases[[i]][,1] 
  a.neurons[[i]] <- sigmoid(z.neurons[[i]])
  
}



  
  

# Backpropagation ---------------------------------------------------------

errors[[kNetworkLength]] <- (a.neurons[[kNetworkLength]] - output.labels) * diff_sigmoid(z.neurons[[kNetworkLength]])


# Backpropagate the error -------------------------------------------------

for(i in kNetworkLength:2){
  
  errors[[i-1]] <- (t(attack.weights[[i]]) %*% errors[[i]]) * diff_sigmoid(z.neurons[[i-1]])
  attack.weights[[i]] <- errors[[i]] %*% t(a.neurons[[i-1]])
  attack.biases[[i]] <- errors[[i]]
  
}

a.neurons[[1]] <- a.neurons[[1]] - 0.5 * (errors[[1]] + (0.5 * (a.neurons[[1]] - input.labels)))

}



plot.output(a.neurons[[3]])


test <- as.data.frame(a.neurons[[1]]) %>%
                        mutate(pixel = 1:784,
                               x_pos = ((pixel - 1) %% 28 + 1),
                               y_pos = abs(29-(((pixel - 1) %/% 28) + 1)))

ggplot(test,
       aes(x = x_pos,
           y = y_pos,
           fill = V1)) +
  geom_tile()



test <- as.data.frame(input.labels) %>%
  mutate(pixel = 1:784,
         x_pos = ((pixel - 1) %% 28 + 1),
         y_pos = abs(29-(((pixel - 1) %/% 28) + 1)))

ggplot(test,
       aes(x = x_pos,
           y = y_pos,
           fill = input.labels)) +
  geom_tile()
