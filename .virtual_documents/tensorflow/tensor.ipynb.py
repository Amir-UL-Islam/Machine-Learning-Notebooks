import tensorflow as tf
import numpy as np


d0 = tf.ones((1, ))


d1 = tf.ones((2, ))


d2 = tf.ones((2, 2))
d2


d3 = tf.ones((2, 2, 2))
print('Ones \n', d3)

d3 = tf.ones_like((2, 2, 2))
print('\n Ones like \n', d3)


tf.constant(3, shape=(2, 2))


credit_numpy = np.array([[ 2.0000e+00,  1.0000e+00,  2.4000e+01,  3.9130e+03],
                        [ 2.0000e+00,  2.0000e+00,  2.6000e+01,  2.6820e+03],
                        [ 2.0000e+00,  2.0000e+00,  3.4000e+01,  2.9239e+04]])

# Creating Constant
tensor_constant = tf.constant(credit_numpy)

tensor_constant.shape


tf.Variable(credit_numpy, dtype=tf.int32)


# Define tensors A1 and A23 as constants
A1 = tf.constant([1, 2, 3, 4])
A23 = tf.constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = tf.ones_like(A1)
B23 = tf.ones_like(A23)

# Perform element-wise multiplication
C1 = tf.multiply(A1, B1)
C23 = tf.multiply(A23, B23)

# Print the tensors C1 and C23
print('\n C1: {}'.format(C1.numpy()))
print('\n C23: {}'.format(C23.numpy()))


# Library
from tensorflow import reduce_sum, reduce_mean, reduce_prod, reduce_max, reduce_min, constant

# Defining Tensor Constant
b1 = constant([[1, 2, 3], [4, 5, 6]])

# Operations
rs = reduce_sum(b1, axis=1, keepdims=True) # here the opperation will go thrugh column wise

rmin_row = reduce_mean(b1, axis=0, keepdims=True)
rmax_column = reduce_mean(b1, axis=1, keepdims=True)

print(rs, '\n')
print('row wise mean \n', rmin_row)
print('\n column wise mean \n', rmax_column)














from tensorflow import constant, matmul, keras

# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features, params)
print(billpred)

# Compute and print the error
error =  bill - billpred

# print(error.numpy())


# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
    return params[0] + feature1*params[1] + feature2*params[2]

# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
    # Set the predicted values
    predictions = linear_regression(params, feature1, feature2)
  
    # Use the mean absolute error loss
    return keras.losses.mae(targets, predictions)

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform minimization and print trainable variables
for j in range(10):
    opt.minimize(lambda: loss_function(params, price_log, size_log, bedrooms), var_list=[params])
    
    print_results(params)


def linear_regression(intercept, slope, feature, y):
    '''
    Defining function for prediction and loss Calculation
    '''
    
    prediction = tf.reduce_sum(slope*feature + intercept, axis=1, keepdims=True)
    loss = keras.losses.mean_squared_error(bill, prediction)
    return print('predictions \n', prediction, '\n loss \n', loss)



linear_regression(params[0], params[1], features, bill)


# Define Loss Function
loss = keras.losses.mean_squared_error(bill, billpred)
print(loss)



# library
import pandas as pd

housing_data = pd.read_csv('datasets/housing.csv')
display(housing_data.info())
housing_data.head()



# Defining slope and intercept
intercept = tf.Variable(0.1, tf.float32)
slope = tf.Variable(0.1, tf.float32)

# model defination
def linear_regression(intercept, slope, features):
    return intercept + slope*features

# loss function
def loss_function(intercept, slope, features, targets):
    predictions = linear_regression(intercept, slope, features)
    return tf.keras.losses.mean_squared_error(targets, predictions)


# Optimizer
opt = tf.keras.optimizers.Adam()

# Loading data in chunk
for batch in pd.read_csv('datasets/housing.csv', chunksize=100): 
    # Extracting features
    total_bedroom = np.array(batch['total_bedrooms'], np.float32)
    
    # Extracting targets
    house_value = np.array(batch['median_house_value'], np.float32)
    
    # Minimize the loss 
    opt.minimize(lambda: loss_function(intercept, slope, total_bedroom, house_value), var_list=[intercept, slope])
    

# Printing Values
print(intercept.numpy(), slope.numpy())


# Define x
x = tf.Variable(-1.0)

'''
In other words, we are doing differenciation in tensorflow 
'''

# Define y with instance  of GradientTape
with tf.GradientTape() as tape:
    tape.watch(x) # rate of change y respect of x.
    y = tf.multiply(x, x)

# Evaluate the gradiant of y at x=-1
gradient = tape.gradient(y, x)
print(gradient.numpy())


x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = 3 * x ** 4
    
gradient = tape.gradient(y, x)
print(gradient.numpy())


# Libray
from tensorflow import reshape, Variable
from skimage import color
import matplotlib.pyplot as plt

# Reading data
color_tensor = plt.imread('datasets/image_data/gabrielle-anwar-og-34387.jpg')
color_tensor = Variable(color_tensor)

# converting to grayscale 
# gray_tensor = color.rgb2gray(color_tensor)

# Reshape the grayscale image tensor into a vector
# gray_vector = reshape(gray_tensor, (784, 1))

# Reshape the color image tensor into a vector
# color_vector = reshape(color_tensor, (2352, 1))







































