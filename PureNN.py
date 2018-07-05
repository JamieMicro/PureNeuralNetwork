#############################################################################################################
#############################################################################################################
# This is a neural network written using for loops and arrays for learning purposes.
# It supports a 3 layered neural network with an input, 1 hidden & output layer.
# If you plan on using your own data then modify the parameter "nnLayers" below based on your data and also
# change the parameter "dataFilePath" to the path of your data file. Lastly, you must update the parameters
# "featureColumns" and "answerColumn" to match the the feature columns in your data.
# Email me at joliver81@gmail.com with any feedback or suggestions. Enjoy!
#############################################################################################################
#############################################################################################################


import numpy as np
import pandas as pd

############################
# Helper Functions
############################
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def feedForward(a, w, b):
    # Process the layer and return the activations for a training example
    num_of_input_neurons = a.shape[0]
    num_of_output_neurons = w.shape[0]

    Z = np.zeros((num_of_output_neurons, 1))    # Result of activation * weight + bias for sigmoid activation function
    A = np.zeros((num_of_output_neurons, 1))    # Resulting activation for next layer

    for n_idx in range(num_of_output_neurons):      # Loop to calculate the activation for each neuron in the layer
        for w_idx in range(num_of_input_neurons):   # Calculate Z from previous layers activation & weights/bias
            Z[n_idx] += (a[w_idx] * w[n_idx][w_idx])

        Z[n_idx] += b[n_idx]            # Add bias
        A[n_idx] = sigmoid(Z[n_idx])    # Calculate sigmoid activation

    return A, Z

def backPropOutput(a, y):
    # Back prop from output layer
    num_of_outputs = y.shape[0]

    d = np.zeros((num_of_outputs, 1))

    for i in range(num_of_outputs):
        d[i] = a[i] - y[i]

    return d.reshape(1, num_of_outputs)

############################
# End Helper Functions
############################


############################
# Begin Main Program
############################

##### Parameters #####
dataFilePath = r'c:\MyProjects\ML_Learning_App\data.txt'  # Path to training data
nnLayers = [2, 5, 3]                                      # NN layers [input, hidden, output]
learning_rate = .001                                      # Set learning rate
number_of_epochs = 50                                     # Set # of epochs (learning iterations)

##### Data Load #####
# Load data with pandas
fullDataset = pd.read_csv(dataFilePath)
featureColumns = ['x1','x2']
answerColumn = ['y']

X = fullDataset[featureColumns]
y = fullDataset[answerColumn]

print('Features X:')
print(X.head())
print('Labels y:')
print(y.head())

##### NN Initialization/Setup #####

# Randomly initialize the weights and bias
theta1 = np.random.rand(nnLayers[1], nnLayers[0])
bias1 = np.zeros((nnLayers[1], 1))
theta2 = np.random.rand(nnLayers[2], nnLayers[1])
bias2 = np.zeros((nnLayers[2], 1))
print('========== Initialized Theta1 ==========')
print(theta1)
print('========== Initialized Theta2 ==========')
print(theta2)

# Create label matrix y
matrix_y = np.zeros((y.size,nnLayers[2]))

# Convert y labels to one hot matrix
for iAnswers in range(y.size):
    matrix_y[iAnswers][y.ix[iAnswers,0]-1] = 1

# For each epoch
for iEpochs in range(number_of_epochs):
    # Network params
    a1_cache = np.empty((0, nnLayers[1]))       # Saved values for back prop
    z1_cache = np.empty((0, nnLayers[1]))       # Saved values for back prop
    a2_cache = np.empty((0, nnLayers[2]))       # Saved values for back prop
    z2_cache = np.empty((0, nnLayers[2]))       # Saved values for back prop
    for i in range(y.size):
        ##################################################
        # Feed forward layer 1
        ##################################################
        a0 = X.loc[i].as_matrix()
        w1 = theta1
        b1 = bias1
        a1, z1 = feedForward(a0, w1, b1)
        a1_cache = np.append(a1_cache, a1.transpose(), axis=0)
        z1_cache = np.append(z1_cache, z1.transpose(), axis=0)
        ##################################################

        ##################################################
        # Feed forward layer 2
        ##################################################
        w2 = theta2
        b2 = bias2
        a2, z2 = feedForward(a1, w2, b2)
        a2_cache = np.append(a2_cache, a2.transpose(), axis=0)
        z2_cache = np.append(z2_cache, z2.transpose(), axis=0)
        ##################################################

    total_cost = 0
    for i in range(y.size):
        ##################################################
        # Back prop
        ##################################################
        dz2 = backPropOutput(a2_cache[i], matrix_y[i])

        # Accumulate the cost
        total_cost += np.sum(dz2)

        # Compute gradients
        dz1 = np.dot(w2.transpose(), dz2.transpose()) * (a1_cache[i] * (1-a1_cache[i])).reshape(nnLayers[1], 1)

        a1_cache_temp = a1_cache[i].reshape(nnLayers[1], 1)

        if i == 0:
            dw2 = np.dot(dz2.T, a1_cache_temp.T)
            db2 = np.sum(dz2.T, axis=1, keepdims=True)
        else:
            dw2 = dw2 + np.dot(dz2.T, a1_cache_temp.T)
            db2 = db2 + np.sum(dz2.T, axis=1, keepdims=True)

        x_temp = X.loc[i].as_matrix().reshape(nnLayers[0], 1)

        if i == 0:
            dw1 = np.dot(dz1, x_temp.T)
            db1 = np.sum(dz1.T, axis=1, keepdims=True)
        else:
            dw1 = dw1 + np.dot(dz1, x_temp.T)
            db1 = db1 + np.sum(dz1.T, axis=1, keepdims=True)
        ##################################################

    ##################################################
    # Update Weights
    ##################################################
    theta2 = theta2-learning_rate*dw2
    bias2 = bias2-learning_rate*db2
    theta1 = theta1-learning_rate*dw1
    bias1 = bias1-learning_rate*db1
    ##################################################

    print('Epoch: ' + str(iEpochs) + ' Cost: ' + str(total_cost))


print('===== Trained Weights/Bias =====')
print('W1:')
print(theta1)
print('b1:')
print(bias1)

print('W2:')
print(theta2)
print('b2:')
print(bias2)


























