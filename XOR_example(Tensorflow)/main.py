import numpy as np
import tensorflow as tf

#=================================================
X = np.array([
    [1,0,1,0],
    [1,1,0,0],
])

Y = np.array([[0,1,1,0]])
#=================================================

m_global = X.shape[1]  # number of training examples m

#=================================================
# Layer Size를 정의
#=================================================
def layer_sizes(X,Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return (n_x, 10, n_y)

#========================================
# Create placeholders X, Y
#========================================
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X') # None : be flexible on the number of examples
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y') # None : be flexible on the number of examples

    return X, Y

#========================================
# Initialize parameters W,b
#========================================
def initialize_parameters(layers_dims):
    parameters = {}         # dictionary
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable("W" + str(l), [layers_dims[l], layers_dims[l - 1]], initializer = tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable("b" + str(l), [layers_dims[l], 1],initializer=tf.zeros_initializer())

    return parameters

with tf.Session() as sess:
    parameters = initialize_parameters(layer_sizes(X,Y))
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))