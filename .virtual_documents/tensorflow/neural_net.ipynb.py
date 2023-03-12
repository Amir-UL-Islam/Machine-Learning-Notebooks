








import pandas as pd
import numpy as np
import tensorflow as tf



data = pd.read_csv('datasets/housing.csv')
data.head()


input_data = tf.constant([data.loc(axis=1)['housing_median_age', 'total_rooms', 'total_bedrooms'].iloc[0].values], tf.float32)
input_data.shape


weights = tf.Variable([[0.2], [0.3], [0.001]])


bias = tf.Variable([0.5])


product = tf.matmul(input_data, weights)
dense = tf.keras.activations.sigmoid(bias + product)












