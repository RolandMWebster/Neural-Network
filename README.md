# NeuralNetworkDigits
This project is around building a neural network 'from scratch'. That is, no packages are loaded to build the model for me, the training process for the model is coded from scratch, this includes the feedforward of inputs and backpropagation of error. 

The purpose of the neural network is to classify the images from the MNIST dataset of handwritten digits 0-9.

I have used the following page to guide me through the model building process:
http://neuralnetworksanddeeplearning.com/

This page supplies all of the learning material required to build the model. It also provides a guide in Python to building your own model, however I have no Python experience so that wasn't much help to me!

I have taken the dataset from here:
http://makeyourownneuralnetwork.blogspot.com/2015/03/the-mnist-dataset-of-handwitten-digits.html

This page provides a .csv file that can be read into R instead of reading in the images directly.

**Files:**
1. Data Prep.R
2. Visualising Data.R
3. Functions.R
4. Model v1.0.R

**Outline:**
1. Read-in .csv data file and prepare it for the model. This includes a train/test split.
2. Plot the images. It helps bridge the gap between a .csv filled with numbers and the images they represent.
3. User defined functions used in the model.
4. The model. This code carries out the training process.
