import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# This makes plots appear in the notebook
get_ipython().run_line_magic("matplotlib", " inline")
sns.set_theme(style="darkgrid")

# getting the data
data = pd.read_csv('datasets/SOCR-HeightWeight.csv', nrows=50)

# Creating figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

sns.regplot(x='Height(Inches)', y='Weight(Pounds)', data=data)
ax.set_xlabel('Linear Regression')

plt.show()


# Reading the Data
sales_df = pd.read_csv('datasets/advertising_and_sales_clean.csv')
sales_df.head()


sales_df.info()


sales_df.describe()


sales_df['influencer'].value_counts()


import numpy as np
from sklearn.model_selection import train_test_split

# Create X from the radio column's values
X = np.array(sales_df['radio'])
# print(X.shape)

# Create y from the sales column's values
y = np.array(sales_df['sales'])

# Reshape X
X = X.reshape(-1, 1)
# print(X.shape)

# # Check the shape of the features and targets
print(X.shape, y.shape)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)


# Make predictions
y_predictions = reg.predict(X_test)

print(predictions[:5])


# Import mean_squared_error
from sklearn.metrics import mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_predictions, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))


from sklearn.model_selection import cross_val_score, KFold

# Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)
cv_results = cv_scores 

# Print scores
print(cv_scores)

# Print the mean
print(np.mean(cv_results))

# Print the standard deviation
print(np.std(cv_results))

# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))


# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(X_train, y_train, color="blue")

# Create line plot
plt.plot(X_test, y_predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()


# import library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Spliting data
# X_train, X_test, y_train, y_test = 

# Encodering 
ohe = OneHotEncoder(sparse=False) # If we don't set (sparse=False). Then It will through an erron
                                  # ValueError: all the input arrays must have same number of dimensions,
                                  # but the array at index 0 has 2 dimension(s) 
                                  # and the array at index 3 has 0 dimension(s)
            
            
                                  # Because by Default OneHotEncoder returns a sparse Matrix
influencer = np.array(sales_df['influencer'].values)
influencer = influencer.reshape(-1, 1)
influencer= ohe.fit_transform(influencer)
# print(influencer.shape)



tv = np.array(sales_df['tv'].values)
tv = tv.reshape(-1, 1)
# print(tv.shape)

radio = np.array(sales_df['radio'].values)
radio = radio.reshape(-1, 1)
# print(radio.shape)

social_media = np.array(sales_df['social_media'].values)
social_media = social_media.reshape(-1, 1)
# print(social_media.shape)

# Scaleing
scaler = StandardScaler()

scaled_tv = scaler.fit_transform(tv)

scaled_radio = scaler.fit_transform(radio)

scaled_social = scaler.fit_transform(social_media)



# concatenate
sales = np.concatenate((scaled_tv, scaled_radio, scaled_social, influencer),axis=1)
print(sales.shape)



# Pipeline




# # Create X and y arrays
# X = sales.drop("sales", axis=1).values
# y = sales["sales"].values


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Instantiate the model
# reg = LinearRegression()

# # Fit the model to the data
# reg.fit(X_train, y_train)

# # Make predictions
# y_pred = reg.predict(X_test)
# print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))


# Getting the data
data = pd.read_csv('datasets/heart.csv')

# Setting figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Ploting the data
plt.plot(sorted(data['Age']))
plt.show()


# Reading the data
chunk_df = pd.read_csv('datasets/telecom_churn_clean.csv')
chunk_df.info()


chunk_df.describe()


# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = chunk_df.drop("churn", axis=1).values
y = chunk_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))


# Create neighbors
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
  
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)
  
    # Fit the model
    knn.fit(X_train, y_train)
  
    # Compute accuracy
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)
    
print('n_neighbors: ', neighbors, '\n')
print('train_accuracies: ', train_accuracies, '\n')
print('test_accuracies: ', test_accuracies)


# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()









