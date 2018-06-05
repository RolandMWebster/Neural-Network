# Define Sigmoid Function -------------------------------------------------

# Define our sigmoid function. This is used to map neuron values onto (0,1)

sigmoid <- function(x){
  1/(1 + exp(-x))
}

# Define our first derivative of the sigmoid function. This is used when calculating the error.
diff_sigmoid <- function(x){
  sigmoid(x) * (1 - sigmoid(x))
}

# FeedForward -------------------------------------------------------------

# We define a function to perform the feedforward process. This calculates the values
# of the neurons, given input weights and biases.

feedForward <- function(x,y,z){
  
  # x: input values
  # y: weights
  # z: bias
  
  # Multiply input by weight
  output <- y %*% x
  # Add Bias
  output + z
  
}

