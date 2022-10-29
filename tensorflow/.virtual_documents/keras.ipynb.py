from tensorflow.keras.layers import Dense, Input
from keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow  as tf


# Defining the input Shape
input_layer = Input(shape=(1, )) # Here the input will be a row of columns(can be 10, 100 or 20)

# Output
output_layer = Dense(1)(input_layer)

# Creating the model
model = Model(input_layer, output_layer)

# Compiling the Model
model.compile(optimizer='adam',
              loss='mae',
              metrics=['accuracy'])

# Model Summary
model.summary()



def show_model_diagram(model):
    img = plot_model(model, to_file='model.png')
    return img

# Calling the Function
show_model_diagram(model)


import pandas as pd
from sklearn.model_selection import train_test_split

games_tourney_train = pd.read_csv('datasets/basketball_data/games_tourney.csv')
display(games_tourney_train.head())
print(games_tourney_train.shape)


X_train, X_test, y_train, y_test = train_test_split(games_tourney_train['seed_diff'],
                                                    games_tourney_train['score_diff'],
                                                    test_size=0.2,
                                                    random_state=0)

# Now fit the model
movements = model.fit(X_train, y_train,
                      epochs=10,
                      validation_split=0.001,
                      verbose=True)


plt.plot(movements.history['accuracy'], color='red', label='train')
plt.plot(movements.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()


# Evaluate the model on the test data
model.evaluate(X_test, y_test, verbose=False)


# Imports
from tensorflow.keras.layers import Embedding, Flatten
from numpy import unique

# Reading data 
games_season = pd.read_csv('datasets/basketball_data/games_season.csv')

# Count the unique number of teams
n_teams = unique(games_season['team_1']).shape[0]

print(n_teams)

# Create an embedding layer
team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')





# Create an input layer for the team ID
teamid_in = Input(shape=(1,))

# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# Combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')


# Load the input layer from tensorflow.keras.layers


# Input layer for team 1
team_in_1 = Input(shape=(1,), name='Team-1-In')

# Separate input layer for team 2
team_in_2 = Input(shape=(1,), name='Team-2-In')


# Lookup team 1 in the team strength model
team_1_strength = team_strength_model(team_in_1)

# Lookup team 2 in the team strength model
team_2_strength = team_strength_model(team_in_2)
print(team_1_strength.is_tensor_like)


# Import the Subtract layer from tensorflow.keras
from tensorflow.keras.layers import Subtract

# Create a subtract layer using the inputs from the previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])

print(score_diff)


# Create the model
model = Model([team_in_1, team_in_2], score_diff)

# Compile the model
model.compile('adam', 'mean_absolute_error')


show_model_diagram(model)


# Get the team_1 column from the regular season data
input_1 = games_season['team_1']

# Get the team_2 column from the regular season data
input_2 = games_season['team_2']

# Fit the model to input 1 and 2, using score diff as a target
model.fit([input_1, input_2],
          games_season['score_diff'],
          epochs=5,
          validation_split=0.1,
          verbose=True)


# Get team_1 from the tournament data
input_1 = games_tourney_train['team_1']

# Get team_2 from the tournament data
input_2 = games_tourney_train['team_2']

# Evaluate the model using these inputs
print(model.evaluate([input_1, input_2], games_tourney_train['score_diff'], verbose=False))


from tensorflow.keras.layers import Concatenate

# Create an Input for each team
team_in_1 = Input(shape=(1,), name='Team-1-In')
team_in_2 = Input(shape=(1,), name='Team-2-In')

# Create an input for home vs away
home_in = Input(shape=(1,), name='Home-In')

# Lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# Combine the team strengths with the home input using a Concatenate layer, then add a Dense layer
out = Concatenate()([team_1_strength, team_2_strength, home_in])
out = Dense(1)(out)


# Import the model class


# Make a Model
model = Model([team_in_1, team_in_2, home_in], out)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])


show_model_diagram(model)


# Fit the model to the games_season dataset
movements = model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],
          games_season['score_diff'],
          epochs=1,
          verbose=True,
          validation_split=0.1)

# Evaluate the model on the games_tourney dataset
model.evaluate([games_season['team_1'], games_season['team_2'], games_season['home']],
                     games_season['score_diff'],
                     verbose=False)


model.summary()


plt.plot(movements.history['accuracy'], color='red', label='train')
plt.plot(movements.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()


games_tourney = pd.read_csv('datasets/basketball_data/games_tourney.csv')
games_tourney.head()


# Predict
games_tourney['pred'] = model.predict([games_tourney['team_1'], games_tourney['team_2'] , games_tourney['home']])
games_tourney.head()


# Create an input layer with 3 columns
input_tensor = Input((3,))

# Pass it to a Dense layer with 1 unit
output_tensor = Dense(1)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


show_model_diagram(model)


# Fit the model
model.fit(games_tourney[['home', 'seed_diff', 'pred']],
          games_tourney['score_diff'],
          epochs=5,
          verbose=True)


# Evaluate the model on the games_tourney_test dataset
print(model.evaluate(games_tourney[['home', 'seed_diff', 'pred']],
               games_tourney['score_diff'], verbose=False))


# Create an input layer with 2 columns
input_tensor = Input(shape=(2, ))

# Create the first output
output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor)

# Create the second output (use the first output as input here)
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False)(output_tensor_1)

# Create a model with 2 outputs
model = Model(input_tensor, [output_tensor_1, output_tensor_2])


show_model_diagram(model)


# Import the Adam optimizer
from tensorflow.keras.optimizers import Adam

# Compile the model with 2 losses and the Adam optimzer with a higher learning rate
model.compile(loss=['mean_absolute_error', 'binary_crossentropy'],optimizer=Adam(learning_rate=0.01))

# Fit the model to the tournament training data, with 2 inputs and 2 outputs
model.fit(games_tourney[['seed_diff', 'pred']],
          [games_tourney[['score_diff']], games_tourney_train[['won']]],
          epochs=10,
          verbose=True,
          batch_size=16384)


print(model.get_weights())


# Import the sigmoid function from scipy
from scipy.special import expit as sigmoid

# Weight from the model
weight = -1.241304

# Print the approximate win probability of a predicted close game
print(sigmoid(1 * weight))

# Print the approximate win probability of a predicted blowout game
print(sigmoid(10 * weight))


# Evaluate the model on new data
print(model.evaluate(games_tourney[['seed_diff', 'pred']],
               [games_tourney[['score_diff']], games_tourney[['won']]], verbose=False))
