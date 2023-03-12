import pandas as pd
import numpy as np


labels = ['shoe', 'shirt', 'shoe', 'shirt', 'dress', 'dress', 'dress']

# The number of image categories
n_categories = 3

# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])

# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))

print(ohe_labels)
# Loop over the labels
for i in range(len(labels)):
    # Find the location of this label in the categories variable
    j = np.where(categories)
    
#     # Set the corresponding zero to one
#     ohe_labels[j] = 1









