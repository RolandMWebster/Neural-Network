# Define Sigmoid Function -------------------------------------------------

# This is used to map neuron values onto (0,1)

sigmoid <- function(x){
  1/(1 + exp(-x))
}

# Define Sigmoid Prime ----------------------------------------------------

# First derivative of the sigmoid function. This is used when calculating the error.
diff_sigmoid <- function(x){
  sigmoid(x) * (1 - sigmoid(x))
}