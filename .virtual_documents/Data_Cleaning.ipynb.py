import pandas as pd
import numpy as np

data = pd.read_csv('datasets/NFL Play by Play 2009-2016 (v3).csv', nrows=100000, dtype='unicode')

display(data.head())
print(data.shape)


display(data.describe(include='all'))


# get the number of missing data points per column
missing_values_count = data.isnull().sum()
print(missing_values_count[0:20])


# how many total missing values do we have?
total_cells = np.product(data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)


# remove all the rows that contain a missing value
data.dropna()


# remove all columns with at least one missing value
columns_with_na_dropped = data.dropna(axis=1)
columns_with_na_dropped.head()


# just how much data did we lose?
print("Columns in original dataset: get_ipython().run_line_magic("d", " \n\" % data.shape[1])")
print("Columns with na's dropped: get_ipython().run_line_magic("d"", " % columns_with_na_dropped.shape[1])")


# replace all NA's with 0
subset_of_data = data.loc[:, 'EPA':'Season']
display(subset_of_data.fillna(0).head(10))



# replace all NA's the value that comes directly after it in the same column, 
# then replace all the remaining na's with 0
subset_of_data.fillna(method='bfill', axis=0).fillna(0).head(10)


# Finding the mean of non zero values
display(subset_of_data['airEPA'].astype(np.float32).fillna(0).mean())

print('\n', 'Filling the Missing one With Mean Value', '\n')
# Filling the data with mean value
display(subset_of_data['airEPA'].fillna(subset_of_data['airEPA'].astype(np.float32).fillna(0).mean()).head(10))














import pandas as pd

# Create series with male and female values
non_categorical_series = pd.Series(['male', 'female', 'male', 'female'])

# Convert the text series to a categorical series
categorical_series = non_categorical_series.astype('category')

# printing the categorical series
print(categorical_series)

# Print the numeric codes for each value
print(categorical_series.cat.codes)

# Print the category names
print(categorical_series.cat.categories)


import pandas as pd

# Create series with male and female values
non_categorical_series = pd.Series(['male', 'female', 'male', 'female'])

# Create dummy or one-hot encoded variables
print(pd.get_dummies(non_categorical_series))




















































































































































































